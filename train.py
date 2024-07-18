import argparse
import json
import os
import pickle
import time
from collections import defaultdict
from contextlib import suppress
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from tabulate import tabulate
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm

from datasets import build_dataloader
from modeling import cfg, build_model
from solver import build_optimizer
from utils.distribute import synchronize, all_gather, is_main_process
from utils.eval import eval_f1, eval_f1_with_boundarys, do_eval
from utils.misc import SmoothedValue, MetricLogger
import torch.backends.cudnn as cudnn
import random, shutil


def make_inputs(inputs, device):
    keys = ['imgs', 'video_path', 'frame_masks']
    results = {}
    if isinstance(inputs, dict):
        for key in keys:
            if key in inputs:
                val = inputs[key]
                if isinstance(val, torch.Tensor):
                    val = val.to(device)
                results[key] = val
    elif isinstance(inputs, list):
        targets = defaultdict(list)
        for item in inputs:
            for key in keys:
                if key in item:
                    val = item[key]
                    targets[key].append(val)

        for key in targets:
            results[key] = torch.stack(targets[key], dim=0).to(device)
    else:
        raise NotImplementedError
    return results


def make_targets(cfg, inputs, device):
    targets = inputs['labels'].to(device)
    return targets


def MSPLoss(features, labels):

    class_num, feature_num = 2, 1
    CoVariance = torch.zeros(class_num, feature_num, feature_num).cuda()
    Ave = torch.zeros(class_num, feature_num).cuda()
    Amount = torch.zeros(class_num).cuda()

    features = features.view(-1, 1)  # (200, 1)
    labels = labels.view(-1, 1)  # (200, 1)

    N = features.size(0)
    C = 2
    A = features.size(1)

    NxCxFeatures = features.view(
        N, 1, A
    ).expand(
        N, C, A
    )

    onehot = torch.zeros(N, C).cuda()

    onehot.scatter_(1, labels.view(-1, 1), 1)

    NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

    features_by_sort = NxCxFeatures.mul(NxCxA_onehot)

    Amount_CxA = NxCxA_onehot.sum(0)
    Amount_CxA[Amount_CxA == 0] = 1

    ave_CxA = features_by_sort.sum(0) / Amount_CxA

    var_temp = features_by_sort - \
                ave_CxA.expand(N, C, A).mul(NxCxA_onehot)

    var_temp_ = torch.bmm(
        var_temp.permute(1, 2, 0),
        var_temp.permute(1, 0, 2)
    ).div(Amount_CxA.view(C, A, 1).expand(C, A, A))

    sum_weight_CV = onehot.sum(0).view(C, 1, 1).expand(C, A, A)

    sum_weight_AV = onehot.sum(0).view(C, 1).expand(C, A)

    weight_CV = sum_weight_CV.div(
        sum_weight_CV + Amount.view(C, 1, 1).expand(C, A, A)
    )
    weight_CV[weight_CV != weight_CV] = 0

    weight_AV = sum_weight_AV.div(
        sum_weight_AV + Amount.view(C, 1).expand(C, A)
    )
    weight_AV[weight_AV != weight_AV] = 0

    additional_CV = weight_CV.mul(1 - weight_CV).mul(
        torch.bmm(
            (Ave - ave_CxA).view(C, A, 1),
            (Ave - ave_CxA).view(C, 1, A)
        )
    )

    CoVariance = (CoVariance.mul(1 - weight_CV) + var_temp_.mul(weight_CV)) + additional_CV
    Ave = (Ave.mul(1 - weight_AV) + ave_CxA.mul(weight_AV))
    Amount += onehot.sum(0)

    c_mean = Ave
    c_cov = CoVariance
    loss = (c_mean[0]-c_mean[1])**2 / ( (c_mean[0]-c_mean[1])**2  + (c_cov[0]**2 + c_cov[1]**2).sqrt() )

    return -loss


def train_one_epoch(cfg, args, model, device, optimizer, data_loader, summary_writer, auto_cast, loss_scaler, epoch):
    model.train()

    start = time.time()

    # class_num, feature_num = 2, 1
    # CoVariance = torch.zeros(class_num, feature_num, feature_num).cuda()
    # Ave = torch.zeros(class_num, feature_num).cuda()
    # Amount = torch.zeros(class_num).cuda()

    for i, inputs in enumerate(data_loader):
        # with torch.autograd.set_detect_anomaly(True):
        # ------------ inputs ----------
        samples = make_inputs(inputs, device)
        targets = make_targets(cfg, inputs, device)
        # ------------ inputs ----------

        model_start = time.time()
        with auto_cast():
            # loss_dict = model(samples, targets)
            # total_loss = sum(loss_dict.values())
            # losses, logits_list, hard_targets_from_th = model(samples, targets)
            losses = model(samples, targets)
            assert len(losses) == len(cfg.MODEL.LOSS_WEIGHT), 'num losses do not match to num loss weight!'
            total_loss = sum([w * loss for w, loss in zip(cfg.MODEL.LOSS_WEIGHT, losses)])
            
            # msp_loss = MSPLoss(logits_list[0], hard_targets_from_th)
            # total_loss = cls_loss + msp_loss * cfg.MODEL.MSP_LOSS_WEIGHT

        # ------------ training operations ----------
        optimizer.zero_grad()
        if cfg.SOLVER.AMPE:
            loss_scaler.scale(total_loss).backward()

            # print('^'*88)
            # for name, param in model.named_parameters():
            #     if param.grad is None:
            #         print(name)

            if cfg.SOLVER.CLIP_GRAD > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.SOLVER.CLIP_GRAD)
            loss_scaler.step(optimizer)
            loss_scaler.update()
        else:
            total_loss.backward()
            if cfg.SOLVER.CLIP_GRAD > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.SOLVER.CLIP_GRAD)
            optimizer.step()
        # ------------ training operations ----------

        # ------------ logging ----------
        if is_main_process():
            summary_writer.global_step += 1

            num_heads = len(cfg.MODEL.HEAD_CHOICE)
            keys = ['head_x{:d}'.format(head+1) for head in cfg.MODEL.HEAD_CHOICE]
            loss_dict = dict(zip(keys, losses))
            summary_writer.update(**loss_dict)

            summary_writer.update(lr=optimizer.param_groups[0]['lr'], total_loss=total_loss,
                                total_time=time.time() - start, model_time=time.time() - model_start)
            start = time.time()

            speed = summary_writer.total_time.avg
            eta = str(timedelta(seconds=int((len(data_loader) - i - 1) * speed)))
            if i % 10 == 0:
                print('Epoch{:02d} ({:04d}/{:04d}): {}, Eta:{}'.format(epoch,
                                                                    i,
                                                                    len(data_loader),
                                                                    str(summary_writer),
                                                                    eta
                                                                    ), flush=True)


@torch.no_grad()
def validate(cfg, args, model, device, data_loader, epoch):
    if cfg.INPUT.END_TO_END:
        return validate_end_to_end(cfg, args, model, device, data_loader, epoch)

    model_pred_dict = {}
    model.eval()
    start_time = time.time()
    num_frames = 0
    for i, inputs in enumerate(tqdm(data_loader, total=len(data_loader))):
        samples = make_inputs(inputs, device)
        num_frames += samples['imgs'].shape[0]

        vids = inputs['vid']
        frame_idxs = inputs['frame_idx']
        scores = model(samples)

        scores = scores.cpu().numpy()
        for vid, frame_idx, score in zip(vids, frame_idxs, scores):
            if vid not in model_pred_dict.keys():
                model_pred_dict[vid] = {}
                model_pred_dict[vid]['frame_idx'] = []
                model_pred_dict[vid]['scores'] = []
            model_pred_dict[vid]['frame_idx'].append(frame_idx)
            model_pred_dict[vid]['scores'].append(score)

    synchronize()
    metrics = {}
    data_list = all_gather(model_pred_dict)
    if not is_main_process():
        metrics['F1'] = 0.00
        return metrics
    total_time = time.time() - start_time
    print('Cost {:.2f}s for evaluating {} videos, {:.4f}s/video'.format(total_time, len(data_loader.dataset), total_time / len(data_loader.dataset)))
    print('{:.9f}ms/frame'.format(total_time * 1000 / num_frames))

    model_pred_dict = defaultdict(dict)
    for p in data_list:
        for vid in p:
            if 'frame_idx' not in model_pred_dict[vid]:
                model_pred_dict[vid]['frame_idx'] = []
                model_pred_dict[vid]['scores'] = []

            model_pred_dict[vid]['frame_idx'].extend(p[vid]['frame_idx'])
            model_pred_dict[vid]['scores'].extend(p[vid]['scores'])

    for vid in model_pred_dict:
        frame_idx = np.array(model_pred_dict[vid]['frame_idx'])
        scores = np.array(model_pred_dict[vid]['scores'])
        _, indices = np.unique(frame_idx, return_index=True)
        frame_idx = frame_idx[indices]
        scores = scores[indices]

        indices = np.argsort(frame_idx)
        model_pred_dict[vid]['frame_idx'] = frame_idx[indices].tolist()
        model_pred_dict[vid]['scores'] = scores[indices].tolist()

    gt_path = f'data/Kinetics-GEBD/k400_mr345_{data_loader.dataset.split}_min_change_duration0.3.pkl'
    f1, rec, prec = eval_f1(model_pred_dict, gt_path, threshold=cfg.TEST.THRESHOLD)
    print('F1: {:.4f}, Rec: {:.4f}, Prec: {:.4f}'.format(f1, rec, prec))
    metrics['F1'] = f1
    metrics['Rec'] = rec
    metrics['Prec'] = prec
    return metrics


@torch.no_grad()
def validate_end_to_end(cfg, args, model, device, data_loader, epoch):
    metrics = {}
    if cfg.TEST.PRED_FILE:
        if not is_main_process():
            return metrics
        with open(cfg.TEST.PRED_FILE, 'rb') as f:
            model_pred_dict = pickle.load(f)
        print(f'Load results from {cfg.TEST.PRED_FILE}.')
    else:
        model_pred_dict = defaultdict(dict)
        model.eval()

        start_time = time.time()
        num_frames = 0

        #**************************************************
        #uncomment the following lines while testing FPS
        # start_idx = 50
        # num_videos = 200
        #**************************************************

        for i, inputs in enumerate(tqdm(data_loader, total=len(data_loader))):
            #**************************************************
            #uncomment the following lines while testing FPS
            # if i < start_idx:
            #     continue
            # if i == start_idx:
            #     torch.cuda.synchronize()
            #     start_time = time.time()
            # if i == start_idx + num_videos:
            #     break
            #**************************************************

            samples = make_inputs(inputs, device)
            num_frames += (samples['imgs'].shape[0] * samples['imgs'].shape[1])

            if cfg.TEST.DYNAMIC:
                outputs = model.dynamic_inference(samples)
                for batch_idx, frame_indices in enumerate(inputs['frame_indices']):
                    vid = inputs['vid'][batch_idx]
                    boundarys = outputs[0][batch_idx]
                    exit_percentages = outputs[1][batch_idx]
                    if 'frame_masks' in inputs:
                        frame_mask = inputs['frame_masks'][batch_idx]
                        frame_indices = frame_indices[frame_mask]
                        boundarys = boundarys[:, frame_mask]
                    model_pred_dict[vid]['frame_idx'] = frame_indices.tolist()
                    model_pred_dict[vid]['boundarys'] = boundarys
                    model_pred_dict[vid]['exit_percentages'] = exit_percentages
            else:
                outputs = model(samples)  # (b, 3, t)
                for batch_idx, frame_indices in enumerate(inputs['frame_indices']):
                    vid = inputs['vid'][batch_idx]
                    scores = outputs[batch_idx]
                    if 'frame_masks' in inputs:
                        frame_mask = inputs['frame_masks'][batch_idx]
                        frame_indices = frame_indices[frame_mask]
                        scores = scores[:, frame_mask]
                    model_pred_dict[vid]['frame_idx'] = frame_indices.tolist()
                    model_pred_dict[vid]['scores'] = [s.tolist() for s in scores]
            
                    if data_loader.dataset.name == 'TAPOS':
                        model_pred_dict[vid]['num_slices'] = inputs['num_slices'][batch_idx]

            # if num_frames >= 10000:
            #     break

        synchronize()
        data_list = all_gather(model_pred_dict)
        if not is_main_process():
            metrics['F1'] = 0.00
            return metrics
        total_time = time.time() - start_time

        #******************************************
        #uncomment the next line while not testing FPS

        num_videos = len(data_loader.dataset)
        #******************************************

        print('Cost {:.2f}s for evaluating {} videos, {:.4f}s/video'.format(total_time, num_videos, total_time / num_videos))
        print('{:.9f}ms/frame'.format(total_time * 1000 / num_frames))

        model_pred_dict = {}
        for p in data_list:
            model_pred_dict.update(p)
    
    num_heads = len(cfg.MODEL.HEAD_CHOICE)
    # gather scores of slices inside the same video
    if data_loader.dataset.name == 'TAPOS':
        model_pred_dict_new = defaultdict(dict)
        vnames = []
        for vid in model_pred_dict.keys():
            vname, _ = vid.split('_slice')
            vnames.append(vname)
        for vname in set(vnames):
            vid = vname + '_slice0'
            if model_pred_dict[vid]['num_slices'] == 1:
                model_pred_dict_new[vname]['frame_idx'] = model_pred_dict[vid]['frame_idx']
                model_pred_dict_new[vname]['scores'] = model_pred_dict[vid]['scores']
            else:
                model_pred_dict_new[vname]['frame_idx'] = []
                scores_ungrouped = []
                for slice in range(model_pred_dict[vid]['num_slices']):
                    vid = vname + f'_slice{slice}'
                    model_pred_dict_new[vname]['frame_idx'].extend(model_pred_dict[vid]['frame_idx'])
                    scores_ungrouped.append(model_pred_dict[vid]['scores'])
                scores = [[] for _ in range(num_heads)]
                for head in range(num_heads):
                    for s in scores_ungrouped:
                        scores[head].extend(s[head])
                model_pred_dict_new[vname]['scores'] = scores
        model_pred_dict = model_pred_dict_new

    if not cfg.TEST.PRED_FILE and data_loader.dataset.split == 'val':
        save_path = os.path.join(args.output_dir, f'model_pred_dict_epoch{epoch:02d}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(model_pred_dict, f)
        print(f'Saved results to {save_path}.')
    
    if data_loader.dataset.name == 'GEBD':
        gt_path = os.path.join('data/Kinetics-GEBD', f'k400_mr345_{data_loader.dataset.split}_min_change_duration0.3.pkl')
    elif data_loader.dataset.name == 'TAPOS':
        gt_path = os.path.join('data/TAPOS', f'tapos_gt_{data_loader.dataset.split}.pkl')
    else:
        raise NotImplemented

    if args.all_thres:
        rel_dis_thres = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    else:
        rel_dis_thres = [0.05]

    if cfg.TEST.DYNAMIC:
        num_heads = 1
        # with open('boundarys.pkl', 'wb') as f:
        #     pickle.dump(model_pred_dict, f)
        results, exit_percentages = eval_f1_with_boundarys(model_pred_dict, gt_path, rel_dis_thres=rel_dis_thres)
        print(f'average exit percentages: head_x2 {exit_percentages[0]}, head_x3 {exit_percentages[1]}')
        flops = 520.59 - 146.63 * exit_percentages[0] - 80.98 * exit_percentages[1]
        print(f'average flops per video: {flops}GMac')
    
    else:
        num_heads = len(cfg.MODEL.HEAD_CHOICE)
        results, pred_dict, exit_dict = eval_f1(model_pred_dict, gt_path,
                                            num_heads=num_heads,
                                            threshold=cfg.TEST.THRESHOLD,
                                            return_pred_dict=True,
                                            rel_dis_thres=rel_dis_thres)
        
        save_path = os.path.join(args.output_dir, f'exit_dict_epoch{epoch:02d}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(exit_dict, f)
        print(f'Saved flitered results to {save_path}.')

    metrics_list = []

    for head in range(num_heads):

        list_rec = []
        list_prec = []
        list_f1 = []

        for th in rel_dis_thres:
            if cfg.TEST.DYNAMIC:
                f1, rec, prec = results[th]
            else:
                f1, rec, prec = results[th][head]
            list_rec.append(rec)
            list_prec.append(prec)
            list_f1.append(f1)

        headers = rel_dis_thres + ['Avg']

        avg_rec = np.mean(list_rec)
        avg_prec = np.mean(list_prec)
        avg_F1 = np.mean(list_f1)

        tabulate_data = [
            ['Recall'] + list_rec + [avg_rec],
            ['Precision'] + list_prec + [avg_prec],
            ['F1'] + list_f1 + [avg_F1],
        ]
        print(f'Results for {data_loader.dataset.name}_{data_loader.dataset.split}:')
        print(tabulate(tabulate_data, headers=headers, floatfmt='.4f'))

        if cfg.TEST.DYNAMIC:
            f1, rec, prec = results[0.05]
        else:
            f1, rec, prec = results[0.05][head]
        print('F1@0.05: {:.4f}, Rec: {:.4f}, Prec: {:.4f}'.format(f1, rec, prec))
        metrics = {}
        metrics['F1'] = f1
        metrics['Rec'] = rec
        metrics['Prec'] = prec
        
        metrics_list.append(metrics)

    return metrics_list


def main(cfg, args):
    train_data_loader = build_dataloader(cfg, args, cfg.DATASETS.TRAIN, is_train=True)
    val_data_loader = build_dataloader(cfg, args, cfg.DATASETS.TEST, is_train=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model(cfg)
    model = model.to(device)


    #**************************************************
    #If test the GFLOPS, uncomment the following lines

    # from fvcore.nn import FlopCountAnalysis
    # x = torch.rand(1,100,3,224,224).cuda()
    # model.eval()
    # flops = FlopCountAnalysis(model, x)
    # print("FLOPs: ", flops.total() / 1e9)
    # return
    #**************************************************


    start_epoch = -1
    if args.resume:
        state_dict = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(state_dict['model'])
        start_epoch = state_dict['epoch']
        if is_main_process():
            print('Loaded from {}, Epoch: {}'.format(args.resume, start_epoch), flush=True)

    # exp_name = '_ann{}_dim{}_window{}_group{}_{}'.format(cfg.INPUT.ANNOTATORS,
    #                                                      cfg.MODEL.DIMENSION,
    #                                                      cfg.MODEL.WINDOW_SIZE,
    #                                                      cfg.MODEL.SIMILARITY_GROUP,
    #                                                      cfg.MODEL.SIMILARITY_FUNC
    #                                                      )
    exp_name = args.expname
    output_dir = cfg.OUTPUT_DIR + exp_name
    os.makedirs(output_dir, exist_ok=True)
    args.output_dir = output_dir

    if not os.path.exists(os.path.join(output_dir, 'scripts')):
        os.makedirs(os.path.join(output_dir, 'scripts'), exist_ok=True)
    saved_py = ['modeling/e2e_model_diff_former.py', 'modeling/diff_former.py', 'modeling/resnet.py', \
                'modeling/baseline.py', 'post_process.py', 'train.py', 'config-files/baseline_end_to_end_diff_former.yaml', \
                'datasets/dataset.py', 'utils/eval.py']
    for py in saved_py:
        dst_py = os.path.join(output_dir, 'scripts', py.split('/')[-1])
        shutil.copyfile(py, dst_py)

    if args.test_only:
        validate(cfg, args, model, device, val_data_loader, start_epoch)
        return

    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

    optimizer = build_optimizer(cfg, [p for p in model.parameters() if p.requires_grad])
    scheduler = MultiStepLR(optimizer, milestones=cfg.SOLVER.MILESTONES)
    # scheduler = CosineAnnealingLR(optimizer, T_max=cfg.SOLVER.MAX_EPOCHS, eta_min=1e-4)
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2)
    if args.resume:
        for name, obj in [('optimizer', optimizer), ('scheduler', scheduler)]:
            if name in state_dict:
                obj.load_state_dict(state_dict[name])
                if is_main_process():
                    print('Loaded {} from {}'.format(name, args.resume), flush=True)

    summary_writer = MetricLogger(log_dir=os.path.join(output_dir, 'logs')) if is_main_process() else None
    if summary_writer is not None:
        summary_writer.add_meter('lr', SmoothedValue(fmt='{value:.5f}'))
        summary_writer.add_meter('total_time', SmoothedValue(fmt='{avg:.3f}s'))
        summary_writer.add_meter('model_time', SmoothedValue(fmt='{avg:.3f}s'))

    auto_cast = torch.cuda.amp.autocast if cfg.SOLVER.AMPE else suppress
    loss_scaler = torch.cuda.amp.GradScaler() if cfg.SOLVER.AMPE else None

    best_f1 = 0.
    for epoch in range(start_epoch + 1, cfg.SOLVER.MAX_EPOCHS):
        train_one_epoch(cfg, args, model, device, optimizer, train_data_loader, summary_writer, auto_cast, loss_scaler, epoch)
        metrics_list = validate(cfg, args, model, device, val_data_loader, epoch)
        # if isinstance(scheduler, ReduceLROnPlateau):
        #     scheduler.step(metrics['F1'])
        # else:
        scheduler.step()

        if is_main_process():
            f1 = metrics_list['F1'] if cfg.TEST.DYNAMIC else metrics_list[-1]['F1']
            save_path = os.path.join(output_dir, 'model_best.pth')
            if f1 > best_f1:
                model_state_dict = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
                # save_path = os.path.join(output_dir, 'model_best.pth')
                torch.save({
                    'model': model_state_dict,
                    'epoch': epoch,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'metrics': metrics_list
                }, save_path)
                best_f1 = f1
            with open(os.path.join(output_dir, 'metrics.txt'), 'a') as f:
                if cfg.TEST.DYNAMIC:
                    content = 'Dynamic inference, F1: {:.4f}, Rec: {:.4f}, Prec: {:.4f}'.format(
                        metrics_list['F1'], metrics_list['Rec'], metrics_list['Prec'])
                else:
                    content = 'Epoch: {:02d},  '.format(epoch) + \
                        ''.join(['Head_x{:d} F1: {:.4f}, Rec: {:.4f}, Prec: {:.4f},  ' \
                                .format(head+1, metrics_list[i]['F1'], metrics_list[i]['Rec'], metrics_list[i]['Prec']) \
                                    for i, head in enumerate(cfg.MODEL.HEAD_CHOICE)]) + '\n'
                f.write(content)
            print('Saved to {}'.format(save_path))


def init_seeds(seed, cuda_deterministic=True):
    cudnn.enabled = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default='config-files/baseline.yaml')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--test-only", action='store_true', default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expname", type=str, default='test')
    parser.add_argument("--all-thres", action='store_true', default=True, help='test using all thresholds [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]')
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.local_rank:
        args.local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
    args.num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = args.num_gpus > 1

    if torch.cuda.is_available():
        # torch.backends.cudnn.benchmark = True
        init_seeds(args.seed + args.local_rank)
        if args.distributed:
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(backend="nccl", world_size=args.num_gpus, rank=args.local_rank)
            dist.barrier()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if is_main_process():
        print('Args: \n{}'.format(args))
        print('Configs: \n{}'.format(cfg))

    main(cfg, args)
