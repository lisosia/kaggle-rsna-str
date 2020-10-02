set -x

# model 035ep1
# calib_p
python3 ./sub.py output/035_pe_present___448/fold0_ep1.pt --post-pe-present-calib-factor 8.555037588568537 --post1-percentile 99.0

