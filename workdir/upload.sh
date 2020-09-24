set -xe

DATANAME=rsna-src

M1=output/001_base/fold0_ep1.pt

COMMENT="sub test. exp001, agg-by-95percentile, other are filled by average"

OUT=../input/$DATANAME/workdir.tar_
rm $OUT

tar --exclude __pycache__ -cvf $OUT conf/ src/ train.py sub.py timm $M1
cd ../input/$DATANAME/
kaggle datasets version -m "${COMMENT}"

