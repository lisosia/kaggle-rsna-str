#!/usr/bin/env python3

import argparse
from collections import defaultdict
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import src.configuration as C
from src.models import get_img_model, ImgModelPE, ImgModel
import src.utils as utils
from src.utils import get_logger
from src.criterion import ImgLoss
from src.datasets import RsnaDatasetTest, RsnaDatasetTest2, RsnaDatasetTest3, RsnaDatasetTest3Valid
from src.postprocess import calib_p
# from train import run_nn

# fix issue: "received 0 items of ancdata"
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
torch.multiprocessing.set_sharing_strategy('file_system')


def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--config", "-c", required=True, help="Config file path")
    parser.add_argument("weight_path", help="weight path")
    parser.add_argument("--debug", action='store_true', help="debug")
    parser.add_argument("--skip", action='store_true', help="skip for commit-time infer")
    parser.add_argument("--validation", action='store_true', help="for validation")
    ### post process
    parser.add_argument("--post-pe-present-calib-factor", required=True, type=float, help="pe_present_on_image calibrtoin (only used for pe_present_on_image)")
    parser.add_argument("--post1-percentile", required=True, type=float, help="postprocess1 of pe_present->exam_pos. used percentile")
    parser.add_argument("--post1-calib-factor", required=True, type=float, help="postprocess1 of pe_present->exam_pos. calib factor")
    return parser.parse_args()
args = get_args()

SEED = 42
DEVICE = "cuda"
def log(msg): print(msg)
def log_w(msg): print(msg)

DATADIR = Path("../input/rsna-str-pulmonary-embolism-detection/")

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
    sample_sub_path = "sample_submission.csv" if not args.validation else "validation/fold0.sample_submission.csv"
    sub = pd.read_csv(DATADIR / sample_sub_path)
    sub['label'] = _MEANS['pe_present_on_image']
    for feat in _MEANS.keys():
        sub.loc[sub.id.str.contains(feat, regex=False), 'label'] = _MEANS[feat]
    return sub


DO_PE_POS_IEFER = False
print("=== DO_PE_POS_IEFER", DO_PE_POS_IEFER)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def one_study_user_stacking(ids, preds_list, z_pos, lgb_models, features):
    # 並び順の事故が怖いので一応idを使ってます
    one_study_test = pd.DataFrame(ids, columns=['id'])
    for pred_n, preds in enumerate(preds_list):
        one_study_test[f'pred{pred_n}'] = preds
    one_study_test['z_pos'] = z_pos
    one_study_test = one_study_test.sort_values('z_pos')
    for pred_n in range(len(preds_list)):
        for i in range(1, 10):
            one_study_test[f'pred{pred_n}_pre{i}'] = one_study_test[f'pred{pred_n}'].shift(i)
            one_study_test[f'pred{pred_n}_post{i}'] = one_study_test[f'pred{pred_n}'].shift(-i)

    # not used now
    # one_study_test[f'pre_z_pos_diff1'] = one_study_test['z_pos'] - one_study_test['z_pos'].shift(1)

    z_pos_max = one_study_test.z_pos.max()
    z_pos_min = one_study_test.z_pos.min()
    one_study_test['z_pos_norm'] = (one_study_test['z_pos'] - z_pos_min) / (z_pos_max - z_pos_min)

    test_preds = np.zeros(len(z_pos))
    ### features = list(set(list(one_study_test)) - set(['id', 'z_pos']))

    if args.debug:
        print("stacking feature:", features)
        print("stacking input\n", one_study_test[features])

    for model in lgb_models:
        test_preds += model.predict(one_study_test[features]) / 5
    one_study_test['stacking_pred'] = sigmoid(test_preds)
    return one_study_test.sort_index()

def main():
    utils.set_seed(SEED)
    device = torch.device(DEVICE)

    # prepare Dataframe first
    TEST_PATH = "test.csv" if not args.validation else "validation/fold0.test.csv"
    df_test = pd.read_csv(DATADIR / TEST_PATH)
    df_sub = load_sub_filled_average()
    df_sub = df_sub.set_index('id')

    # sanity check
    for _std in df_test.StudyInstanceUID.unique():
        assert (len(df_sub.loc[_std + '_negative_exam_for_pe']) == 1)
    for _sop in df_test.SOPInstanceUID.unique():
        assert (len(df_sub.loc[_sop]) == 1)
    print("sanity check passed. df_test, df_sub prepared")

    print(len(df_test))
    if args.skip and len(df_test) == 146853:
        print("=== Only public test is vissible. Skip inference and just submit mean-prediction!")
        df_sub.reset_index(inplace=True)
        df_sub.to_csv("submission.csv", index=False)
        return


    # ### model 001 : image-level pred of pe_present_on_image
    model = ImgModel(archi="efficientnet_b0", pretrained=False).to(device)

    # lgb_models = [pickle.load(open(f'lgb_models/lgb_fold{i}.pkl', 'rb')) for i in range(5)]
    lgb_models = [pickle.load(open(f'lgb_models/exp035_1018/lgb_seed0_fold{i}.pkl', 'rb')) for i in range(5)]
    features = ['pred0', 'pred0_post1', 'pred0_post2', 'pred0_post3', 'pred0_post4', 'pred0_post5', 'pred0_post6', 'pred0_post7', 'pred0_post8', 'pred0_post9', 'pred0_pre1', 'pred0_pre2', 'pred0_pre3', 'pred0_pre4', 'pred0_pre5', 'pred0_pre6', 'pred0_pre7', 'pred0_pre8', 'pred0_pre9', 'z_pos_norm']

    # result_pe = sub_2(None, model, args.weight_path)
    ### model 010: pe_present+right,left,center
    # model = ImgModelPE(archi="efficientnet_b0", pretrained=False).to(device)

    log(f"Model type: {model.__class__.__name__}")
    result_pe = sub_4(None, model, args.weight_path, valid_df=df_test if args.validation else None)
    utils.save_pickle(result_pe, 'cache/xxx.pickle')

    for study in df_test.StudyInstanceUID.unique():
        res = result_pe[study]
        res_out = res["outputs"]

        # pe_present_on_image
        for sop, pred in zip(res["ids"], res_out["pe_present_on_image"]):
            df_sub.loc[sop, 'label'] = calib_p(pred, factor=args.post_pe_present_calib_factor)

        preds_list = []
        preds_list.append(
                calib_p( res_out["pe_present_on_image"], factor=5.72045 )  # TODO: calib for each fold
            )
        stacking_pred_df = one_study_user_stacking(res["ids"], preds_list, res['z_pos'], lgb_models, features)
        df_sub.loc[stacking_pred_df['id'], 'label'] = stacking_pred_df['stacking_pred'].values

        if DO_PE_POS_IEFER:
            # agg for right,left,center. pe_present-weighted average
            # in real-labe, always, P(right) < P(pe_present)
            # if predicted P(right) ~= P(pe_present) for most slices, then ave_right ~= 1
            # below agg is euqal to `p(pe_present)`-weighted `p(right)/p(pe_present)`
            pe_present_prob_sum = np.sum(res_out["pe_present_on_image"])
            ave_right  = np.clip( np.sum(res_out["rightsided_pe"]) / pe_present_prob_sum, 0, 1)  #clipping so that average(right) < ave(pe_present)
            ave_left   = np.clip( np.sum(res_out["leftsided_pe" ]) / pe_present_prob_sum, 0, 1)
            ave_center = np.clip( np.sum(res_out["central_pe"]   ) / pe_present_prob_sum, 0, 1)
            # postprocess like sqrt(p) may be good to push-up right/left/central probs for tackle half-right-slice,half-left-slice exam

        ### fill exam_type (negative, indeterminate, positive)
        # pos_exam_prob = np.power(np.mean(res["outputs"] ** 7), 1/7)
        pos_exam_prob = np.percentile(
                calib_p( res_out["pe_present_on_image"], factor=args.post1_calib_factor),
                q=args.post1_percentile
            )

        print("pos_exam_prob", pos_exam_prob, "max_pe_present_prob", np.max(res_out["pe_present_on_image"]))
        # print("DEBUG:", study, pos_exam_prob, "RLC", ave_right, ave_left, ave_center)

        indeterminate_prob = _MEANS['indeterminate']  # from average
        df_sub.loc[study + '_' + 'indeterminate', 'label'] = indeterminate_prob
        df_sub.loc[study + '_' + 'negative_exam_for_pe', 'label'] = (1 - indeterminate_prob) * (1 - pos_exam_prob)

        ### fill right,left,central
        if DO_PE_POS_IEFER:
            pos_exam_prob_real = (1 - indeterminate_prob) * pos_exam_prob
            df_sub.loc[study + '_' + "rightsided_pe", 'label'] = pos_exam_prob_real * ave_right
            df_sub.loc[study + '_' + "leftsided_pe" , 'label'] = pos_exam_prob_real * ave_left
            df_sub.loc[study + '_' + "central_pe"   , 'label'] = pos_exam_prob_real * ave_center
        else:
            for k in ['leftsided_pe', 'rightsided_pe', 'central_pe']:
                df_sub.loc[study + '_' + k, 'label'] = pos_exam_prob * _MEANS_POS[k] + (1 - pos_exam_prob) * _MEANS_NOT_POS[k]


        # for k in ['rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1', 'leftsided_pe', 'rightsided_pe', 'central_pe', 'chronic_pe', 'acute_and_chronic_pe']:
        # fill by pos_prob*ave_pos+(1-pos_prob)*ave_not_pos
        for k in ['rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1', 'chronic_pe', 'acute_and_chronic_pe']:
            df_sub.loc[study + '_' + k, 'label'] = pos_exam_prob * _MEANS_POS[k] + (1 - pos_exam_prob) * _MEANS_NOT_POS[k]

        # import pdb; pdb.set_trace()

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
        if args.debug: break

    print("per study result's keys(): ", result_all[study_id].keys())
    # import pdb; pdb.set_trace()
    return result_all

# sub 3: pe_present + right,left,center
def sub_3(cfg, model, weight_path):
    """
    Returns: 
    result_all["study_id"] -> {
        "outputs" -> {"col_name1" -> np.ndarray, "col_name2 -> np.ndarray}
        "ids -> sop_id_arr
    }
    """
    utils.load_model(weight_path, model)
    result_all = {}
    model = model.eval()
    dataset_sub  = RsnaDatasetTest2()
    dataloader = DataLoader(dataset_sub, batch_size=1, shuffle=False, num_workers=1, collate_fn=lambda x:x)
    for (item) in tqdm(dataloader):
        imgs, study_id, sop_arr = item[0]
        _bs = 64
        outputs_all = defaultdict(list)
        for i in np.arange(0, len(sop_arr), step=_bs):
            _imgs = torch.from_numpy(imgs[i: i+_bs]).cuda()
            with torch.no_grad():
                outputs = model(_imgs)
            for _k in outputs.keys():  # iter over output keys:
                outputs_all[_k].extend(torch.sigmoid(outputs[_k]).cpu().numpy())  # currently all output is binarty logit
        result_all[study_id] = {
            "outputs": dict([(k, np.array(v)) for k, v in outputs_all.items()]),
            "ids": np.array(sop_arr),
        }
        if args.debug: break

    print("per study result's keys(): ", result_all[study_id].keys())
    return result_all

# almost same as sub_3. performance tuning and slightly difference return structure
def sub_4(cfg, model, weight_path, valid_df=None):
    """
    Returns: 
    result_all["study_id"] -> {
        "outputs" -> {"col_name1" -> np.ndarray, "col_name2 -> np.ndarray}
        "ids -> sop_id_arr
    }
    """
    utils.load_model(weight_path, model)
    model = model.eval()
    if valid_df is None:
        dataset_sub  = RsnaDatasetTest3()
    else:
        print("prepare dataset for local validation")
        dataset_sub  = RsnaDatasetTest3Valid(valid_df)

    dataloader = DataLoader(dataset_sub, batch_size=64, shuffle=False, num_workers=2)
    outputs_all = defaultdict(list)
    study_ids = []
    sop_ids = []
    z_positions = []
    debug_cnt = 0
    for imgs, study_arr, sop_arr, z_pos_arr in tqdm(dataloader):
        # import pdb; pdb.set_trace()
        imgs = imgs.cuda()
        with torch.no_grad():
            # for i in range(10): # simulate 10 models
            for i in range(1):  # TODO do multi model inference
                outputs = model(imgs)
        for _k in outputs.keys():  # iter over output keys:
            outputs_all[_k].extend(torch.sigmoid(outputs[_k]).cpu().numpy())  # currently all output is binarty logit
        study_ids.extend(study_arr)
        sop_ids.extend(sop_arr)
        z_positions.extend(z_pos_arr)
        if args.debug:
            debug_cnt += 1
            if debug_cnt > 6: break

    # gather results per study
    all_output_keys = outputs_all.keys()
    study_ids = np.array(study_ids)
    sop_ids = np.array(sop_ids)
    z_positions = np.array(z_positions)
    result_final = {}
    for study in np.unique(study_ids):
        indice = study_ids == study
        sops = sop_ids[indice]
        z_pos = z_positions[indice]
        result_final[study] = {
            "outputs": dict(
                [(key, np.array(outputs_all[key])[indice]) for key in all_output_keys]
            ),
            "ids": sops,
            'z_pos': z_pos
        }

    print("per study result's keys(): ", result_final[study].keys())
    return result_final

if __name__ == "__main__":
    main()
