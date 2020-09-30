# calib pe_repsent_on_image of exp001ep1-q95-post0
# factor3.825 from validation

import pandas as pd

FACTOR=3.825
def calib_p(arr, factor):  # set factor>1 to enhance positive prob
    return arr * factor / (arr * factor + (1-arr))

PATH="exp001ep1-q95-post0"
df = pd.read_csv(PATH)

feats_not_change = [
 'negative_exam_for_pe',
 'rv_lv_ratio_gte_1',
 'rv_lv_ratio_lt_1',
 'leftsided_pe',
 'chronic_pe',
 'rightsided_pe',
 'acute_and_chronic_pe',
 'central_pe',
 'indeterminate']

bools = df.id.str.contains('negative_exam_for_pe', regex=False).values
for feat in feats_not_change:
    t = df.id.str.contains(feat, regex=False)
    print(feat, t.sum())
    bools = bools | t

targets = ~ bools
print(targets.sum())

df.loc[targets, 'label'] = calib_p(df[targets].label, factor=FACTOR)

df.to_csv("exp001ep1-q95-post0___calibed_3p835.csv", index=False)
