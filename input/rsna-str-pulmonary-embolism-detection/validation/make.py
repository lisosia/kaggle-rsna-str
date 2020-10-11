import pandas as pd
import numpy as np

train = pd.read_csv("train.csv")
split = pd.read_csv("split.csv")
train = train.merge(split, on="StudyInstanceUID")
for f in range(5):
    # make dummy test.csv
    d = train[train.fold == f]
    d[["StudyInstanceUID","SeriesInstanceUID","SOPInstanceUID"]].to_csv(f"validation/fold{f}.test.csv", index=False)

    ids = []
    for key in ["negative_exam_for_pe","rv_lv_ratio_gte_1","rv_lv_ratio_lt_1", 
              "leftsided_pe","chronic_pe","rightsided_pe","acute_and_chronic_pe",
              "central_pe", "indeterminate"]:
        for study in d.StudyInstanceUID.unique():
            ids.append(study + '_' + key)

    for sop in d.SOPInstanceUID.unique():
        ids.append(sop)
    
    pd.DataFrame({
            'id': ids, 
            'label': [0.5]*len(ids)
        }).to_csv(f"validation/fold{f}.sample_submission.csv", index=False)
