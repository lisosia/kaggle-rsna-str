#!/usr/bin/env python3
"""Local Validatoin"""

import sys
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

EXAM_COLS = [
    'negative_exam_for_pe',
    'indeterminate',
    'chronic_pe',
    'acute_and_chronic_pe',
    'central_pe',
    'leftsided_pe',
    'rightsided_pe',
    'rv_lv_ratio_gte_1',
    'rv_lv_ratio_lt_1'
]
WEIGHTS = {
    'negative_exam_for_pe': 0.0736196319,
    'indeterminate': 0.09202453988,
    'chronic_pe': 0.1042944785,
    'acute_and_chronic_pe': 0.1042944785,
    'central_pe': 0.1877300613,
    'leftsided_pe': 0.06257668712,
    'rightsided_pe': 0.06257668712,
    'rv_lv_ratio_gte_1': 0.2346625767,
    'rv_lv_ratio_lt_1': 0.0782208589
}
WEIGHT_IMAGE = 0.07361963

def prep_label(fold=0):
    label_df = pd.read_csv(f'../input/rsna-str-pulmonary-embolism-detection/train.csv')
    label_df = label_df.merge(pd.read_csv('../input/rsna-str-pulmonary-embolism-detection/split.csv'), on='StudyInstanceUID')
    label_df = label_df.merge(pd.read_csv('../input/rsna-str-pulmonary-embolism-detection/study_pos_portion.csv'), on='StudyInstanceUID')
    label_df = label_df[label_df.fold == fold]
    train = pd.read_csv(f'../input/rsna-str-pulmonary-embolism-detection/train.csv')
    agg_study = train.groupby('StudyInstanceUID').first()

    pdata = []
    studies = label_df.StudyInstanceUID.unique()
    for key in EXAM_COLS:
        for study in studies:
            pdata.append({
                'id': study + '_' + key,
                # 'id': study + '_' + key,
                'label': agg_study.loc[study][key]
            })
    
    for c in label_df.itertuples():
        pdata.append({
            'id': c.SOPInstanceUID,
            'label': c.pe_present_on_image
        })

    return pd.DataFrame(pdata), label_df


def split_df(df: pd.DataFrame):
    df = df.copy()
    ret = {}
    is_exam_row = np.zeros(len(df), dtype=bool)
    for key in EXAM_COLS:
        bool_arr = df.id.str.contains(key, regex=False)
        ret[key] = df.loc[bool_arr]
        is_exam_row |= bool_arr
    ret['pe_present_on_image'] = df.loc[ ~ is_exam_row]

    return ret


def calc_score(labels: dict, preds: dict, df_portion: pd.DataFrame, studies):
    """df_portion, SOPInstanceUID->pe_present_portion"""
    df_portion = df_portion.set_index("SOPInstanceUID")
    raw_result = {}
    weight_result = {}
    
    for feat in EXAM_COLS:
        _ind = [std + '_' + feat for std in studies]
        l = labels[feat].set_index('id').loc[_ind]
        p = preds [feat].set_index('id').loc[_ind]
        logloss = log_loss(y_true=l, y_pred=p, normalize=False) * WEIGHTS[feat]  # normalize==False to calc sum, not mean
        # raw_result[feat] = logloss
        weight_result[feat] =  logloss

    # image-level
    sops = labels['pe_present_on_image']['id']
    sops_weights = df_portion.loc[sops]['pe_present_portion']
    # print(sops_weights)
    _label = labels['pe_present_on_image'].set_index("id").loc[sops].values
    _pred  = preds ['pe_present_on_image'].set_index("id").loc[sops].values
    # import pdb; pdb.set_trace()
    logloss_img = log_loss(y_true=_label, y_pred=_pred, sample_weight=sops_weights, normalize=False) * WEIGHT_IMAGE
    weight_result['pe_present_on_image'] = logloss_img

    # agg score
    score = 0.0
    for key, val in weight_result.items():
        score += val
    for key, val in weight_result.items():
        print(f'{key:20s} {val:04.4f} {100*val/score:.4f}%')

    weight_sum = np.sum(list(WEIGHTS.values())) * len(studies) +  np.sum(WEIGHT_IMAGE * sops_weights.values)
    total_score = score / weight_sum

    return total_score


def eval(pred_path, fold=0):
    # label_df = pd.read_csv(f'../input/rsna-str-pulmonary-embolism-detection/validation/fold{fold}.test.csv')
    # label_df.merge(pd.read_csv('../input/rsna-str-pulmonary-embolism-detection/split.csv'), on='StudyInstanceUID')

    label_df, label_df_all = prep_label(fold)
    pred_df = pd.read_csv(pred_path)
    if False:  # debug
        pred_df = label_df.copy()
        pred_df.label = 0.5  # debug

    spl_label = split_df(label_df)
    spl_pred  = split_df(pred_df)

    studies = label_df_all.StudyInstanceUID.unique()
    score = calc_score(spl_label, spl_pred, label_df_all, studies)
    print("\n===> final score", score)
    # import pdb; pdb.set_trace()

if __name__ == "__main__":
    pred_path = sys.argv[1]
    eval(pred_path, fold=0)