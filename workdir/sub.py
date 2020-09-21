#!/usr/bin/env python3

import argparse
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.optim
from torch.utils.data import DataLoader
from apex import amp
from tqdm.auto import tqdm

import src.configuration as C
from src.models import get_img_model, ImgModel
import src.utils as utils
from src.utils import get_logger
from src.criterion import ImgLoss
from src.datasets import RsnaDatasetTest, RsnaDatasetTest2

from train import run_nn


def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--config", "-c", required=True, help="Config file path")
    parser.add_argument("weight_path", help="weight path")
    return parser.parse_args()
args = get_args()

SEED = 42
DEVICE = "cuda"
def log(msg): print(msg)
def log_w(msg): print(msg)


DATADIR = Path("/kaggle/input/rsna-str-pulmonary-embolism-detection/")
_MEANS = {
    'pe_present_on_image': 0.053915069524414806,
    'negative_exam_for_pe': 0.6763928618101033,
    'rv_lv_ratio_gte_1': 0.12875001256566257,
    'rv_lv_ratio_lt_1': 0.17437230326919448,
    'leftsided_pe': 0.21089872969528548,
    'chronic_pe': 0.040139752506710064,
    'rightsided_pe': 0.2575653665766779,
    'acute_and_chronic_pe': 0.019458347341720122,
    'central_pe': 0.054468517151291695,
    'indeterminate': 0.020484822355039723}
_MEANS_POS = {
    'pe_present_on_image': 0.17786572188168448,
    'negative_exam_for_pe': 0.0,
    'rv_lv_ratio_gte_1': 0.4247460706119915,
    'rv_lv_ratio_lt_1': 0.5752539293880086,
    'leftsided_pe': 0.6957545475146886,
    'chronic_pe': 0.13242097466878175,
    'rightsided_pe': 0.849707702540123,
    'acute_and_chronic_pe': 0.0641930545038497,
    'central_pe': 0.1796915446534345,
    'indeterminate': 0.0}
_MEANS_NOT_POS = {
    'pe_present_on_image': 0.0,
    'negative_exam_for_pe': 0.9706048524432512,
    'rv_lv_ratio_gte_1': 0.0,
    'rv_lv_ratio_lt_1': 0.0,
    'leftsided_pe': 0.0,
    'chronic_pe': 0.0,
    'rightsided_pe': 0.0,
    'acute_and_chronic_pe': 0.0,
    'central_pe': 0.0,
    'indeterminate': 0.029395147556748744}

def load_sub_filled_average():
    sub = pd.read_csv(DATADIR / "sample_submission.csv")
    sub['label'] = _MEANS['pe_present_on_image']
    for feat in _MEANS.keys():
        sub.loc[sub.id.str.contains(feat, regex=False), 'label'] = _MEANS[feat]
    return sub


def main():
    utils.set_seed(SEED)
    device = torch.device(DEVICE)

    # model 001 : image-level pred of pe_present_on_image
    model = ImgModel(archi="efficientnet_b0").to(device)
    log(f"Model type: {model.__class__.__name__}")
    result_pe = sub_2(None, model, args.weight_path)
    # result_pe = sub(None, model, args.weight_path)
    cache_file = 'cache/001.pickle'
    with open(cache_file, 'wb') as f:
        pickle.dump(result_pe, f)

    # df_sub = pd.read_csv(DATADIR / "sample_submission.csv")
    df_sub = load_sub_filled_average()
    df_sub = df_sub.set_index('id')
    df_test = pd.read_csv(DATADIR / "test.csv")

    for study in df_test.StudyInstanceUID.unique():
        res = result_pe[study]
        # pe_present_on_image
        for sop, pred in zip(res["ids"], res["outputs"]):
            df_sub.loc[sop, 'label'] = pred

        # fill exam_type (negative, indeterminate, positive)
        pos_exam_prob = np.power(np.mean(res["outputs"] ** 7), 1/7)
        print("pos_exam_prob", pos_exam_prob, "max_pe_present_prob", np.max(res["outputs"]))
        indeterminate_prob = _MEANS['indeterminate']  # from average
        df_sub.loc[study + '_' + 'indeterminate', 'label'] = indeterminate_prob
        df_sub.loc[study + '_' + 'negative_exam_for_pe', 'label'] = (1 - indeterminate_prob) * (1 - pos_exam_prob)

        # fill by pos_prob*ave_pos+(1-pos_prob)*ave_not_pos
        for k in ['rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1', 'leftsided_pe', 'rightsided_pe', 'central_pe', 'chronic_pe', 'acute_and_chronic_pe']:
            df_sub.loc[study + '_' + k, 'label'] = pos_exam_prob * _MEANS_POS[k] + (1 - pos_exam_prob) * _MEANS_NOT_POS[k]

    df_sub.reset_index(inplace=True)
    df_sub.to_csv("submission.csv", index=False)


def sub(cfg, model, weight_path):
    utils.load_model(weight_path, model)
    model = model.eval()
    dataset_sub  = RsnaDatasetTest()

    df_test = pd.read_csv(DATADIR / "test.csv")
    result_all = {}
    for study in tqdm(df_test.StudyInstanceUID.unique()):
        dataset_sub.set_StudyInstanceUID(study)
        loader_sub = DataLoader(dataset_sub, batch_size=32, shuffle=False, pin_memory=True, num_workers=6)
        with torch.no_grad():
            result = run_nn(cfg, 'test', model, loader_sub)
            result_all[study] = result

    print("per study result's keys(): ", result_all[study].keys())
    return result_all


def sub_2(cfg, model, weight_path):
    utils.load_model(weight_path, model)
    result_all = {}
    model = model.eval()
    dataset_sub  = RsnaDatasetTest2()
    dataloader = DataLoader(dataset_sub, batch_size=1, shuffle=False, num_workers=3, collate_fn=lambda x:x)
    for (item) in tqdm(dataloader):
        imgs, study_id, sop_arr = item[0]
        _bs = 128
        preds = []
        for i in np.arange(0, len(sop_arr), step=_bs):
            _imgs = torch.from_numpy(imgs[i: i+_bs]).cuda()
            with torch.no_grad():
                outputs = model(_imgs)
            res = torch.sigmoid(outputs["pe_present_on_image"]).cpu().numpy()
            preds.extend(res)
        result_all[study_id] = {
            "outputs": np.array(preds),
            "ids": np.array(sop_arr),
        }

    print("per study result's keys(): ", result_all[study_id].keys())
    # import pdb; pdb.set_trace()
    return result_all


if __name__ == "__main__":
    main()
