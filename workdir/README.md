## exp

#### PE and PE-position
001 
image level data. `pe_present_on_image` のみ train
effnet-b0 ep1(0origin)が最大. それ以降はloss上昇. accもほぼ低下
一応,lr=1e-3 * 0.5 で finetuneしたが同様にoverfit
ep1のmetric:
    acc_pe_present_on_image:0.9628  pre_pe_present_on_image:0.7161  rec_pe_present_on_image:0.5576  f1_pe_present_on_image:0.6270  logloss_pe_present_on_image:0.1179

010
left,right,centerをsub-taskで学習
epo6, [best] ep:0 loss:0.1924
    acc_pe_present_on_image:0.9650  acc_rightsided_pe:0.9653  acc_leftsided_pe:0.9658  acc_central_pe:0.9805  
    pre_pe_present_on_image:0.8836  rec_pe_present_on_image:0.4342  f1_pe_present_on_image:0.5822  
    pre_rightsided_pe:0.9151  rec_rightsided_pe:0.3803  f1_rightsided_pe:0.5373  
    pre_leftsided_pe:0.9093  rec_leftsided_pe:0.3488  f1_leftsided_pe:0.5042  
    pre_central_pe:0.6773  rec_central_pe:0.0414  f1_central_pe:0.0781  
    logloss_pe_present_on_image:0.1129  logloss_rightsided_pe:0.1111  logloss_leftsided_pe:0.1093  logloss_central_pe:0.0588
    score:0.1129 
[比較対象]: image-level. ex. [C] => pe_presentかつcenter_peをpositiveとしたときの, mean-predictionのlogloss
    [R] percentage 0.050096, mean-pred-logloss 0.198797
    [L] percentage 0.046010, mean-pred-logloss 0.186596
    [C] percentage 0.017059, mean-pred-logloss 0.086362
    [PE_PREESNT] percentage 0.053915, mean-pred-logloss 0.209885
        => mean-predより良い.

postprocesesのpercentile. modelやepochに敏感 notebook/eda-postprocess.ipynb
epoが進むと over-confidenceになっている
soft-label したら緩和するかも

■ わかったこと 09/29
right,left,center予想の影響は, 現状モデルだととても小さい
    exp010 pe pos q98.9   LB.317  　　※percentile98.9でbestなのはvalidationと一致. validationは信頼して良いかも
    right,left,centerをexpected-meanでfill   LB.320
    ＝＞改善は .003
exp010(pe_presentのみinfer) に関しては epo0 << epo1
    LB .320 => .305
    ＝＞ .015 違う. しかし validation logloss は ep1のほうが小さい epo0=.1129 epo1=.2079
さらに, pe_present以外のすべての列を exp001ep1 (LB0.264) と同じにしても
    LB .305 => .302
    .264との差は .038
        ＝＝＝＞ pe_present がめちゃくちゃ重要
        　　　　 にもかかわらず、local validation と一致していない？
        　　　　　　　＝＝＝＞わかった。 image-level eval は pe_presentの量に Weightがある !  これが違いだと思われる
　　　　　　　　　　　　　　　＝＞ POSITIVE-EXAMについてのみ, pe_presentが正確なら良い！！！！！！！！
　　　　　　　　　　　　　　　　   　　ちなみに、しかもpos-portionが高いものは2乗で重要 (portion2倍だと, rowが2倍 & q_i が2倍)
                                        portionが多い方が post-processで工夫しやすいはず (ex. gauss-ぼかしとか) なのでそちらも重要そう

001
[!] calib ＝＞ wegiht-logloss=0.2688

010 
pe_posも学習
    ep0(LB悪かったやつ)   logloss_pe_present_weighted:0.4605        f1_pe_present_on_image:0.5822  logloss_pe_present_on_image:0.1129
[!] calib ＝＞ weight-loglsos=0.2641

030
001と同じ. やりなおしただけ
    ep0 logloss_pe_present_weighted:0.6336
    ep1 logloss_pe_present_weighted:0.3405 
    ep2 logloss_pe_present_weighted:0.4968
031
ovesample=4する  (randomCropのみ)
    ep0: logloss_pe_present_weighted:0.4270
    ep1: logloss_pe_present_weighted:0.5990
[!] calibすると ?

031___tune
resume:exp001,ep1   
get_transform_v2 ->   softlabel(eps,e-2), oversample=3, low-lr 2e-4, v,hhlip
    ep1 logloss_pe_present_weighted:0.3584
    ep2 logloss_pe_present_weighted:0.3255        f1_pe_present_on_image:0.6327  logloss_pe_present_on_image:0.1150
    ep3 logloss_pe_present_weighted:0.4064
    ep4 Overfitなので切り上げ
    ★すこしゆっくりになった. 少し良くなったかもだが, タイミングだけの問題かも
    ap_pe_present_on_image:0.6885
[!] calib ＝＞ 0.26535   (factor=3.825)

032 augv3
    だめ. 年のため soflabelけす,lr戻してfinetuneしたがoverfitはさけれらない

035 448サイズ(512)  ep0だけapex(ep1以降はnan発生したのでFP32)
    ep1 ave_loss:0.090737 ap_pe_present_on_image:0.7751    f1_pe_present_on_image:0.6830
    ep2 AP低下, loss上昇, したep3~切り上げ
ep1 calib ＝＞ 0.2344 (factor=8.5550)

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


