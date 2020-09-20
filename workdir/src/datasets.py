from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch.utils.data as data
import albumentations as alb

DATADIR = Path("/kaggle/input/rsna-str-pulmonary-embolism-detection/")

_EXAM_TYPES = ["negative", "indeterminate", "positive"]
def _encode_exam_type(row): return _EXAM_TYPES.index(row["exam_type"])
def _decode_exam_type(idx): return _EXAM_TYPES[idx]
_PE_TYPES = ["chronic", "acute_chronic", "acute"]
def _encode_pe_type(row): return _PE_TYPES.index(row["pe_type"])
def _decode_pe_type(idx): return _PE_TYPES[idx]

def rawlabel_to_label(row):
    return dict({
        "exam_type": _encode_exam_type(row),
        "pe_present_on_image": row["pe_present_on_image"],  # can be is only when exam_type positive
        "pe_type": _encode_pe_type(row),                    # valid only when exam_type=True and pe_present_on_image
        "rv_lv_ratio_gte_1": row["rv_lv_ratio_gte_1"],      # valid only when exam_type=True and pe_present_on_image
        "rightsided_pe": row["rightsided_pe"],  # valid only when exam_type=True and pe_present_on_image, not-exclusive
        "leftsided_pe": row["leftsided_pe"],    # valid only when exam_type=True and pe_present_on_image, not-exclusive
        "central_pe": row["central_pe"],        # valid only when exam_type=True and pe_present_on_image, not-exclusive
        "qa_motion": row["qa_motion"],     # could be 1 when and only when indeterminate
        "qa_contrast": row["qa_contrast"], # could be 1 when and only when indeterminate
        "flow_artifact": row["flow_artifact"],  # [optional] always valid
        "true_filling_defect_not_pe": row["true_filling_defect_not_pe"],  # [optional] valid when exam_type positive or negative
    })

class RsnaDataset(data.Dataset):
    """Image Level Dataset"""
    def __init__(self, fold, phase):
        assert phase in ["train", "valid"]
        self.phase = phase
        self.datafir = DATADIR
        # prepare df
        df_train = pd.read_csv(DATADIR / "train.csv")
        df_fold = pd.read_csv(DATADIR / "split.csv")
        df_prefix = pd.read_csv(DATADIR / "sop_to_prefix.csv")
        df = df_train.merge(df_fold, on="StudyInstanceUID")  # fold row
        df = df_train.merge(df_prefix, on="StudyInstanceUID")  # img_prefix row
        if phase == "train":
            self.df = df[df.fold != fold]
        elif phase == "valid":
            self.df = df[df.fold == fold]
        self.transform = get_transform_v1()

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int):
        sample = self.df.iloc[idx, :]
        image = get_img_jpg256(sample)
        if self.phase == "train":
            image = self.transform(image=image)["image"]
        label = rawlabel_to_label(sample)
        return {
            "image": image, 
            "label": label
        }


def get_img_jpg256(r):
    folder = DATADIR / "train-jpegs" / r["StudyInstanceUID"] / r["StudyInstanceUID"]
    img_path = folder / r["img_prefix"] + r["SOPInstanceUID"]
    return cv2.imread(img_path)

def get_transform_v1():
    return alb.Compose([
        alb.RandomCrop(224, 244, p=1)
    ])

# Split CSV
def split_stratified(outpath):
    from sklearn.model_selection import StratifiedKFold
    import seaborn as sns
    import matplotlib.pyplot as plt

    df = pd.read_csv(DATADIR / "train.csv")
    df_study = df.groupby("StudyInstanceUID").first()
    df_study = df_study.reset_index()

    # for exam lebel, 3 pattern: [pos, indeterminant, neg]
    def to_exam_type(negative, indeterminant):
        if negative == 1 and indeterminant == 0:
            return "negative"
        elif negative == 0 and indeterminant == 1:
            return "indeterminate"
        elif negative == 0 and indeterminant == 0:
            return "positive"
        else:
            raise Exception("unknown type")

    df_study["exam_type"] = [to_exam_type(a, b) for a,b in zip(df_study.negative_exam_for_pe, df_study.indeterminate)]
    _exam_type =  df_study.exam_type.unique()
    print(f"exam_type: {_exam_type}")
    assert len(_exam_type) == 3

    print(df_study.head())
    studies = df_study.StudyInstanceUID

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    df_study["fold"] = -1
    for fold, (train_idx, valid_idx) in enumerate(skf.split(df_study.StudyInstanceUID, df_study.exam_type)):
        df_study.loc[valid_idx, "fold"] = fold
    
    sns.countplot(data=df_study, x="fold", hue="exam_type")
    plt.show()
    
    df_study[["StudyInstanceUID", "exam_type", "fold"]].to_csv(outpath, index=False)


if __name__ == "__main__":
    outpath = DATADIR / "split.csv"
    split_stratified(outpath)