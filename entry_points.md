## 1st stage train

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

## 2nd stage train

```
<yama>
指示をお願いします。
```

## submit

```
python3 sub_b3_monai_position_1026_ensemble.py
```
