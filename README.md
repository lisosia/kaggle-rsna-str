#ARCHIVE CONTENTS
models              : original models
kaggle-rsna-str     : original codes

#HARDWARE: (The following specs were used to create the original solution)
Intel Xeon Gold 6148（27.5M Cache, 2.40 GHz, 20 Core）×2
NVIDIA Tesla V100（SXM2）
CentOS 7.5

#SOFTWARE (python packages are detailed separately in `requirements.txt`):
python 3.6.5
cuda 9.1.85.3
cudnn 7.0.5
NVRM version: NVIDIA UNIX x86_64 Kernel Module  440.33.01
GCC version:  gcc version 4.8.5 20150623 (Red Hat 4.8.5-28)

#DATA SETUP (assumes the [Kaggle API](https://github.com/Kaggle/kaggle-api) is installed)
# below are the shell commands used in each step, as run from the top level directory
First unzip the code in models_and_code.tgz or download the code from github.
Next, set the kaggle datas for the competition in the input folder.

```
cd input/rsna-str-pulmonary-embolism-detection
kaggle competitions download -c rsna-str-pulmonary-embolism-detection
```

(However, you can download the train image data from the following link, so as for the train image, you don't need to download the official kaggle data.)
https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/discussion/185742
This data must also be set to the another folder.

```
cd .. # move from rsna-str-pulmonary-embolism-detection to input
mkdir train-jpegs-512
and set train-images to train-jpegs-512 folder.
```

And external table data is here ( https://www.kaggle.com/yujiariyasu/df-pairs?select=train_with_position.pkl )

```
cd ../input/external_table_data
and set train_with_position.pkl to external_table_data.
```

#SETTINGS.json
NOTE:
SETTINGS.json is in kaggle-rsna-str/workdir folder.

#MODEL BUILD:
First, create 1st stage models and oof. oof is used for training 2nd stage lightgbm models.

```
cd workdir
python3 train.py train conf/final_image_level.yml -o output/b3/oof_fold0.pkl --fold 0
python3 train.py train conf/final_image_level.yml -o output/b3/oof_fold1.pkl --fold 1
python3 train.py train conf/final_image_level.yml -o output/b3/oof_fold2.pkl --fold 2
python3 train.py train conf/final_image_level.yml -o output/b3/oof_fold3.pkl --fold 3
python3 train.py train conf/final_image_level.yml -o output/b3/oof_fold4.pkl --fold 4

python3 train.py train conf/final_position.yml -o output/position/oof_fold0.pkl --fold 0
python3 train.py train conf/final_position.yml -o output/position/oof_fold1.pkl --fold 1
python3 train.py train conf/final_position.yml -o output/position/oof_fold2.pkl --fold 2
python3 train.py train conf/final_position.yml -o output/position/oof_fold3.pkl --fold 3
python3 train.py train conf/final_position.yml -o output/position/oof_fold4.pkl --fold 4
```

<yama>

```
スタッキングモデルの学習について、ここにお願いします。
```

#IF YOU USE OUR TRAINED MODEL
<yama>
モデルのセット等、ここに指示を書いていただければ。
解凍したのち、modelsというフォルダからxxxで始まるファイルをoutput_yujiへ、yyyで始まるファイルを5foldmonaiへ...等お願いします。下にコマンド書いていただければ。

```
cd workdir
mkdir output_yuji
mkdir -p output_jan/5foldmonai
mv xxx* output_yuji/
mv yyy* output_jan/5foldmonai/
```

#SUBMIT:

```
python3 sub_b3_monai_position_1026_ensemble.py
```
