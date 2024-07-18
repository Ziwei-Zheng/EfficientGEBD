# ******************testing BasicGEBD-ResNet50-L1**************************
torchrun --nproc_per_node 8 \
--master_port 1111 train.py \
--expname test \
--test-only \
--resume output/Kinetics-GEBD/x1_r50_basic/model_best.pth \
MODEL.BACKBONE.NAME 'resnet50' \
MODEL.CAT_PREV False \
MODEL.FPN_START_IDX 0 \
MODEL.HEAD_CHOICE [0] \
MODEL.IS_BASIC True
#**************************************************************************