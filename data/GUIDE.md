# GUIDE for preparing input data

## **Kinetics GEBD**
Here is the guide for Kinetics GEBD, you can also refer to [instructions of LOVEU Challenge](https://github.com/StanLei52/GEBD/blob/main/INSTRUCTIONS.md) for help. 

**1-a**. Download [Kinetics-GEBD annotation](https://drive.google.com/drive/folders/1AlPr63Q9D-HAGc5bOUNTzjCiWOC1a3xo): `k400_train_raw_annotation.pkl` and `k400_val_raw_annotation.pkl` in the following path: `EfficientGEBD/data/Kinetics-GEBD/`.

**1-b**. Download videos listed in the [Kinetics-GEBD annotation](https://drive.google.com/drive/folders/15nvSPpogyCYaPCdd4bNw5AkWVsb_kYYn?usp=drive_link) in the following path:`EfficientGEBD/data/Kinetics-GEBD/videos/`
Note that videos in the Kinetics-GEBD dataset are a subset of Kinetics-400 dataset. You can either download the whole [Kinetics-400 dataset](https://drive.google.com/drive/folders/15nvSPpogyCYaPCdd4bNw5AkWVsb_kYYn?usp=drive_link) or just download the part in Kinetics-GEBD dataset.

**1-c**. Trim the videos according to the Kinetics-400 annotations. E.g., after you downloading `3y1V7BNNBds.mp4`, trim this video into a 10-second video `3y1V7BNNBds_000000_000010.mp4` from 0s to 10s in the original video. Note that the start time and end time for each video can be found at the Kinetics-400 annotations.

**1-d**. Extract frames. For example, use the following script to extract frames of video M99mgKKmPqs.
```
ffmpeg -i $video_root/3y1V7BNNBds_000000_000010.mp4 -f image2 -qscale:v 2 -loglevel quiet $frame_root/3y1V7BNNBds_000000_000010/frame%d.png
```

**1-e**. Generate GT files `k400_mr345_train_min_change_duration0.3.pkl` and `k400_mr345_val_min_change_duration0.3.pkl` . Refer to [prepare_k400_release.ipynb](https://github.com/StanLei52/GEBD/blob/main/data/export/prepare_k400_release.ipynb) for generating the GT files. Specifically, you should prepare the train and val split:

generate_frameidx_from_raw(split='train')
generate_frameidx_from_raw(split='val')

To reproduce EfficientGEBD result, you can directly use [my GT files](https://drive.google.com/drive/folders/10daNvdsW1phKg9POh_gGhXQctel_eF3t?usp=sharing) in the following path:`EfficientGEBD/data/Kinetics-GEBD/`(prepare_k400_release.ipynb was [changed](https://github.com/StanLei52/GEBD/issues/3) in 2022.5, the new annotation may lead to a slighly different result).

## **TAPOS**
**2-a**. Download [TAPOS_annotation](https://drive.google.com/file/d/1zdr7FmZpCyg-wNU9TXy0sVnEYgwGrsXw/view?usp=drive_link): `tapos_annotation.json` in the following path:`EfficientGEBD/data/TAPOS/`.

**2-b**. Download videos listed in the  [TAPOS_annotation](https://drive.google.com/drive/folders/1AlPr63Q9D-HAGc5bOUNTzjCiWOC1a3xo) in the following path:`EfficientGEBD/data/TAPOS/videos/`, in which yMK2zxDDs2A is the video ID of YouTube.

**2-c**. Trim the videos according to the TAPOS annotations. E.g., after you downloading `I2Mgku_IMOE.mp4`, trim this video into a 8.214-second video `3y1V7BNNBds_s00027_2_705_10_919.mp4` from 438.321s to 449.816s in the original video via the following command:
```
ffmpeg -i I2Mgku_IMOE.mp4 -ss 438.321 -t 11.495 -c copy 3y1V7BNNBds_s00027_2_705_10_919.mp4
```
 Note that the start time and end time for each video can be found at the `tapos_annotation.json`.
 
Here we provide  `data/TAPOS/trimvideo.py` to trim all the videos via the following command:
```
python data/TAPOS/trimvideo.py
```

**2-d**. Extract frames. The specific script is shown in Kinetics-GEBD. You can download them from [OpenDataLab](https://opendatalab.com/OpenDataLab/TAPOS/tree/main).

You can also use our `data/TAPOS/frame_extract.py` to extract frames from the trimed videos via the following command:
```
python data/TAPOS/frame_extract.py
```

**2-e**.Generate the original GT files `tapos_gt_train.pkl` and `tapos_gt_val.pkl`.  To reproduce EfficientGEBD result, you can directly use our GT files in `data/TAPOS/`

**2-f**. Generate the downsampled GT file. The generated code for this section is in `datasets/dataset.py`, which will automatically run and generate downsampled GT file during training.


## **SoccerNet**

**3-a**. You can follow [SoccerNetv2-DevKit](https://github.com/SilvioGiancola/SoccerNetv2-DevKit?tab=readme-ov-file) to download videos and annotations. To install the pip package simply run:
`pip install SoccerNet`
Make sure the video is in the following path:`/EffSoccerNet/data/videos/`. Note that our task is Task2-CameraShotSegmentation, make sure download `Labels-cameras.json` in each video. 
You can download videos and annotaitons via the following command:
```
python EffSoccerNet/DownloadSoccerNet.py
```
Noting that if you want to download the videos, you will need to fill a  [NDA](https://www.soccer-net.org/) to get the password.

**3-b**. Using `EfficientGEBD/EffSoccerNet/extract_feats.py` to extract and compute features from videos via the following command:
```
python EffSoccerNet/extract_feats.py
```
 Note that we use offline training here, we change the type of model in `EfficientGEBD/EffSoccerNet/extract_feats.py` to generate different pre-extracted and pre-computed features. 
 
 [SoccerNetv2-DevKit](https://github.com/SilvioGiancola/SoccerNetv2-DevKit?tab=readme-ov-file) only provide pre-computed features from ResNet152, we also provide pre-computed features from ResNet50 and CSN, you can  [download]() and place them in `/EffSoccerNet/data/`.

**3-c**.  Generate dataset.  The generated code for this section is in `EffSoccerNet/dataset.py`. While training, choose `--SoccerNet_path`and`--features` for Corresponding pretrained model  in `EffSoccerNet_train.sh` to generate dataset.
