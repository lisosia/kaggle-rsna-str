set -xe

DATANAME=rsna-src

M1=output/010_pe_pos/fold0_ep0.pt

COMMENT="010 pe_present+pe_position. pe_pre-agg 95p, position-agg by sum(probs_right)/sum(probs_pe_pre), set percentile for postprocess"

OUT=../input/$DATANAME/workdir.tar_
rm $OUT

tar --exclude __pycache__ -cvf $OUT conf/ src/ train.py sub.py timm $M1
cd ../input/$DATANAME/
kaggle datasets version -m "${COMMENT}"

