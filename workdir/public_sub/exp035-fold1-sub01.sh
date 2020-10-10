set -x

# model 035ep1
# calib_p

# pe_present_on_image
./sub.py output/035_pe_present___448___apex___resume/fold1_ep1.pt --post-pe-present-calib-factor 5.72045 --post1-percentile 99 --post1-calib-factor 0.5 
mv -i submission.csv public_sub/exp035-fold1-ep1-calib.csv
### pe_present
# raw_pred_035 fold1 ep1 (5.720451550292213, 0.2508630943769819)
### pe_exam
# fold0の0.30よりわるいが..
# best, per percnetile:99 calib:0.5 best_loss 0.3310889169604432   <=== BEST

