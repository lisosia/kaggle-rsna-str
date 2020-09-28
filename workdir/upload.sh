set -xe

DATANAME=rsna-src

M1=output/010_pe_pos/fold0_ep0.pt

COMMENT="[FOR PUBLIC ONLY]: exp010 pe_present only "

OUT=../input/$DATANAME/workdir.tar_
rm $OUT

tar --exclude __pycache__ -cvf $OUT conf/ src/ train.py sub.py public_sub/ timm $M1
cd ../input/$DATANAME/
kaggle datasets version -m "${COMMENT}"

