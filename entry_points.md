## 1st stage train

train 2D-CNN
```
cd workdir
# train 1
for fold in $(seq 0 5) ; do
    python3 train.py train conf/final_image_level.yml -o output/final_image_level/oof_fold${fold}.pkl --fold ${fold}
done
# calculate calibration value
for fold in $(seq 0 5) ; do
    python3 src/oof_opt.py ${fold} output/final_image_level/oof_fold${fold}.pkl
done

# train 2
for fold in $(seq 0 5) ; do
    python3 train.py train conf/final_position.yml -o output/final_position/oof_fold${fold}_pos.pkl --fold ${fold}
    python3 train.py  test conf/final_position.yml -o output/final_position/oof_fold${fold}.pkl     --fold ${fold}
done
```

train 3D-CNN (MONAI).
```
train3DMonai.ipynb
```

## 2nd stage train

Use below notebooks. Note that you need to edit some lines because refactored codes are not fully tested. Check README.md for detail.
```
notebook/stacking_yuji_b3.ipynb
notebook/stacking_yuji_b3_monai_acute_position.ipynb
```

## submit

```
python3 sub_b3_monai_position_1026_ensemble.py
```
