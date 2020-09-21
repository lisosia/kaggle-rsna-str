##### exp
001 
image level data. `pe_present_on_image` のみ train
effnet-b0 epo1が最大. それ以降はloss上昇. accもほぼ低下
一応,lr=1e-3 * 0.5 で finetuneしたが同様にoverfit
 
###### some reference codes
code structrure and trainer loop
https://github.com/appian42/kaggle-rsna-intracranial-hemorrhage/blob/master/src/cnn/
https://github.com/appian42/kaggle-rsna-intracranial-hemorrhage/blob/master/src/cnn/main.py


