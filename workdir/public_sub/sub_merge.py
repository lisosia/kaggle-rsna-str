BASE = "exp001ep1-q95-post0"
FILL_PE_PRE = "exp010ep1-q97-post0"

import pandas as pd

base = pd.read_csv(BASE)
fill_pe_pre = pd.read_csv(FILL_PE_PRE)

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

for feat in feats_not_change:
    print(feat)
    fill_pe_pre.loc[fill_pe_pre.id.str.contains(feat, regex=False), 'label'] = base.loc[base.id.str.contains(feat, regex=False), 'label']

fill_pe_pre.to_csv("out.csv", index=False)

