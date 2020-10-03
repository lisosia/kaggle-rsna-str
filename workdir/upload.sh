set -xe

DATANAME=rsna-src

M1=output/035_pe_present___448/fold0_ep1.pt

COMMENT="[FOR PUBLIC ONLY]: exp035 sub, calip_p() for pe_present, perentile for exam_pos/neg, other is ave-filled"

OUT=../input/$DATANAME/workdir.tar_
rm $OUT

tar --exclude __pycache__ -cvf $OUT conf/ src/ train.py sub.py public_sub/ timm $M1
cd ../input/$DATANAME/
kaggle datasets version -m "${COMMENT}"

