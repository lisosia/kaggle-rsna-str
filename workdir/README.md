## exp

#### PE and PE-position
001 
image level data. `pe_present_on_image` のみ train
effnet-b0 ep1(0origin)が最大. それ以降はloss上昇. accもほぼ低下
一応,lr=1e-3 * 0.5 で finetuneしたが同様にoverfit
ep1のmetric:
    acc_pe_present_on_image:0.9628  pre_pe_present_on_image:0.7161  rec_pe_present_on_image:0.5576  f1_pe_present_on_image:0.6270  logloss_pe_present_on_image:0.1179


#### indeterminate
baseline:
   indeterminate単体. mean-prediction logloss => 0.099920  ※`p*log(p)+(1-p)*log(1-p)`
   すべて0.2でpredictしたときの, 今のcriterionでのtotal-loss=> 0.3375

* 002_indeterminate/
effnet_b0, 3window, img-level
5epo [best] ep:0 total-loss:0.4336

* 002_indeterminate_3_shallow
浅い&high-reso重視の2d-CNN, 3window
[2poch] ep:0 totalloss:0.2969 <===current best
   ※細かいmetric未実装
    epo:1  total-loss=0.305876, logloss_indeterminate:0.0904  logloss_qa_contrast:0.0805  logloss_qa_motion:0.0446 
    mean-prediction のlogloss(0.098)よりすこしだけよい
      eps=1e-3でprobをclip => loglossすべて全く変わらず
      eps=1e-2でprobをclip => ほんのすこし悪くなる(logloss_indeterminate:0.0905)

* 004_indeterminate_3d_pretrain.yml
pretrainがないのが悪いのではと思って,
3dresnet18 のpretrainを一応試したが、epo1がかなり悪かったので切り上げ
epo2まで, [best] ep:2 loss:0.3560

- conf/003_indeterminate_3d(___restart).yml
3dresnet10, PE-Windowのみ,
15epo, [best] ep:11 loss:0.2961
metrics:
{'pre_indeterminate': 1.0, 'rec_indeterminate': 0.03225806451612903, 'f1_indeterminate': 0.0625, 'pre_qa_contrast': 1.0, 'rec_qa_contrast': 0.038461538461538464, 'f1_qa_contrast': 0.07407407407407407, 'pre_qa_motion': 0.0, 'rec_qa_motion': 0.0, 'f1_qa_motion': 0.0}
=> PE-window,256size で qa_motion を判定するのは不可能
=> f1_contrast はepo11 で f1=0.07 あたり
=> ただし、f1は揺れ幅が大きい 
   epo7で 'f1_qa_contrast': 0.23529411764705882が最大だが, 0のときもある
   => indeterminate, contrast, motoin の los_lossを見るべき

#### 考察
- indeterminate prediction

img-levelだとaverage-fillよりloss悪い. f1=0.09も絶望的に悪い
画像を確認した. 
qa_constrast=1の画像は PE-windowで見た時に、心臓(やその他の部位)において、
最も明るい箇所がv=255になっていない.
    discussion情報) PEは白色がくすんでいるの判断することから、qa_constrastは白がたりないことを意味しているだろう
　　centric-PE検査には、心臓部をつかう. left,right も同様だろう
関係ない部位をみても意味がないばかりか、学習に悪影響
-> 3D conv が必要.

下記画像のように ~\ のような形で心臓血管を視認できるのは 8slice/200slice くらい
https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/discussion/183850

心臓位置
目視した： 156 / 241, 109 / 191, 63 / 96, 115 / 169
2/3あたりにある. cropしてもいいけど危険ダヨネ

- qa_motionについては、調査中 [TODO]
https://www.slideshare.net/DukeHeartCenter/ct-imaging-for-cardiac-disease-disease-identification-preprocedural-planning-and-hemodynamic-assessment p7-p9
このような motion artifact は実際の画像でみあたらなかった..

"ct image motion artifact" で検索
https://pubs.rsna.org/doi/10.1148/rg.246045065
qa_motionはshadingとしてあらわれるらしい.

http://rad-base.com/?p=1001

loss:
oversampleきかず


###### some reference codes
code structrure and trainer loop
https://github.com/appian42/kaggle-rsna-intracranial-hemorrhage/blob/master/src/cnn/
https://github.com/appian42/kaggle-rsna-intracranial-hemorrhage/blob/master/src/cnn/main.py


