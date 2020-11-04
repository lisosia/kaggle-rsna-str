## ARCHIVE CONTENTS
models              : original models  
kaggle-rsna-str     : original codes  

## HARDWARE:
Intel Xeon Gold 6148（27.5M Cache, 2.40 GHz, 20 Core）×2  
NVIDIA Tesla V100（SXM2）  
CentOS 7.5  

## SOFTWARE
python 3.6.5  
cuda 9.1.85.3  
cudnn 7.0.5  
NVRM version: NVIDIA UNIX x86_64 Kernel Module  440.33.01  
GCC version:  gcc version 4.8.5 20150623 (Red Hat 4.8.5-28)  

## DATA SETUP
(assumes the [Kaggle API](https://github.com/Kaggle/kaggle-api) is installed)  
below are the shell commands used in each step, as run from the top level directory  
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

## SETTINGS.json
NOTE:  
SETTINGS.json is in kaggle-rsna-str/workdir folder.

## MODEL BUILD:
First, create 1st stage models and oof. oof is used for training 2nd stage lightgbm models.  

#### 1st stage

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

#### 2nd stage

a) calibration for `final_image_level` model  

use below command to get calibration factor `f` for each model
```
python3 src/oof_opt.py <FOLD_NUM> <OOF PICKLE FILE>
```

b) stacking for `pe_present_on_image` using `pe_present_on_image` model

Edit pickle path and calibration factor (`In [6]` cell).  
Then run `/notebook/stacking_yuji_b3.ipynb`.  
Move created models in `lgbs/lgb_seed0_fold{i}.pkl` to `lgb_models/exp035_1018/`

c) stacking for other targets using `pe_preesnt_on_image` model, `position` model, monai model

Edit pickle path, calibration factor and oof csv file path (`In [6]`, `In [138]`, `In [136]`cell). Also Edit oof-file loading function if needed.  
Then Run `notebook/stacking_yuji_b3_monai_acute_position.ipynb`.  
Move created models in `lgbs/{TargetName}_monai_lgb_seed0_fold{i}.pkl` to `lgb_models/b3_exams_monai_position_1026/`.

## IF YOU USE OUR TRAINED MODEL

```
cd workdir
mkdir -p output_yuji output_jan
# unzip model to /datapath/ then:
mv /datapath/models/b3_non_weight output_yuji/
mv /datapath/models/position output_yuji/
mv /datapath/models/5foldmonai output_jan/
```
Note that stacking models are already commited to `workdir/lgb_models/exp035_1018/`, `workdir/lgb_models/b3_exams_monai_position_1026/`.

## SUBMIT:

```
python3 sub_b3_monai_position_1026_ensemble.py
```
