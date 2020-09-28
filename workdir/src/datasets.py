import glob
import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch.utils.data as data
import albumentations as alb
import pydicom
from scipy.ndimage.interpolation import zoom
from src.utils import timer
import src.transforms as transforms

DATADIR = Path("/kaggle/input/rsna-str-pulmonary-embolism-detection/")

_EXAM_TYPES = ["negative", "indeterminate", "positive"]
def _encode_exam_type(row): return _EXAM_TYPES.index(row["exam_type"])
def _decode_exam_type(idx): return _EXAM_TYPES[idx]
_PE_TYPES = ["chronic", "acute_and_chronic", "acute"]
def _encode_pe_type(row): return _PE_TYPES.index(row["pe_type"])
def _decode_pe_type(idx): return _PE_TYPES[idx]

def rawlabel_to_label(row) -> dict:
    ret = dict({
        "exam_type": _encode_exam_type(row),
            "indeterminate": row["indeterminate"],  # dup
        "pe_present_on_image": row["pe_present_on_image"],  # can be is only when exam_type positive
        # "pe_type": _encode_pe_type(row),                    # valid only when exam_type=True and pe_present_on_image
        "chronic_pe":           row["chronic_pe"] * row["pe_present_on_image"],
        "acute_and_chronic_pe": row["acute_and_chronic_pe"] * row["pe_present_on_image"],
        "acute_pe":             (1-row["chronic_pe"]) * (1-row["acute_and_chronic_pe"]) * row["pe_present_on_image"],  # note: not in train.csv

        "rv_lv_ratio_gte_1": row["rv_lv_ratio_gte_1"],      # valid only when exam_type=True and pe_present_on_image
        "rightsided_pe": row["rightsided_pe"] * row["pe_present_on_image"],  # valid only when exam_type=True and pe_present_on_image, not-exclusive
        "leftsided_pe":  row["leftsided_pe"]  * row["pe_present_on_image"],  # valid only when exam_type=True and pe_present_on_image, not-exclusive
        "central_pe":    row["central_pe"]    * row["pe_present_on_image"],  # valid only when exam_type=True and pe_present_on_image, not-exclusive
        "qa_motion": row["qa_motion"],     # could be 1 when and only when indeterminate
        "qa_contrast": row["qa_contrast"], # could be 1 when and only when indeterminate
        "flow_artifact": row["flow_artifact"],  # [optional] always valid
        "true_filling_defect_not_pe": row["true_filling_defect_not_pe"],  # [optional] valid when exam_type positive or negative
    })
    # if multi position has pe, each slice may have one individual pe (left or right or center)
    # so use soft-label.
    if ret["pe_present_on_image"] > 0.5 and (ret["rightsided_pe"]+ret["leftsided_pe"]+ret["central_pe"]) > 1:
        EPS=0.1
        ret["rightsided_pe"] = np.clip(ret["rightsided_pe"], 0, 1-EPS)
        ret["leftsided_pe"] = np.clip(ret["leftsided_pe"], 0, 1-EPS)
        ret["central_pe"] = np.clip(ret["central_pe"], 0, 1-EPS)
    return ret

def rawlabel_to_label_study_level(row) -> dict:
    return dict({
        "exam_type": _encode_exam_type(row),
            "indeterminate": row["indeterminate"],  # dup
        # "pe_type": _encode_pe_type(row),                    # valid only when exam_type=True and pe_present_on_image
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
        df = pd.read_csv(DATADIR / "train.csv")
        df_fold = pd.read_csv(DATADIR / "split.csv")
        df = df.merge(df_fold, on="StudyInstanceUID")  # fold row
        df_prefix = pd.read_csv(DATADIR / "sop_to_prefix.csv")
        df = df.merge(df_prefix, on="SOPInstanceUID")  # img_prefix row
        if phase == "train":
            self.df = df[df.fold != fold]
            self.transform = get_transform_v1()
        elif phase == "valid":
            self.df = df[df.fold == fold]
            self.transform = get_transform_valid_v1()
        # self.df = self.df.iloc[:200]  # debug

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int):
        sample = self.df.iloc[idx, :]
        # label
        label = rawlabel_to_label(sample)
        # image
        image = get_img_jpg256(sample)
        if self.phase == "train":
            image = self.transform(image=image)["image"]
        image = (image.astype(np.float32) / 255).transpose(2,0,1)

        return image, label, sample.SOPInstanceUID


class RsnaDataset3D(data.Dataset):
    """Study(Series) Level Dataset"""
    def __init__(self, fold, phase, oversample=False):
        assert phase in ["train", "valid"]
        self.phase = phase
        self.datafir = DATADIR
        self.oversample = oversample and phase=="train"
        # prepare df
        df = pd.read_csv(DATADIR / "train.csv")
        df_fold = pd.read_csv(DATADIR / "split.csv")
        df = df.merge(df_fold, on="StudyInstanceUID")  # fold row
        df_prefix = pd.read_csv(DATADIR / "sop_to_prefix.csv")
        df = df.merge(df_prefix, on="SOPInstanceUID")  # img_prefix row
        if phase == "train":
            self.df = df[df.fold != fold]
            self.transform = transforms.trans_ind_3d
        elif phase == "valid":
            self.df = df[df.fold == fold]
            self.transform = transforms.trans_ind_3d_valid
        # self.df = self.df.iloc[:20000]  # debug
        self.studies = self.df.StudyInstanceUID.unique()

        if self.oversample:  # oversample for indeterminate
            print("RsnaDataset3D indeterminate oversampled")
            studies_df = self.df.groupby("StudyInstanceUID").first().reset_index()  # study->each-col
            self.ind_num = studies_df.indeterminate.sum()
            self.not_ind_num = len(studies_df) - self.ind_num
            assert self.ind_num + self.not_ind_num == len(studies_df)
            assert self.ind_num < self.not_ind_num  # confirm indeterminate is minor
            self.df_ind = studies_df[studies_df.indeterminate == 1]
            self.df_not_ind = studies_df[studies_df.indeterminate == 0]

    def __len__(self):
        if self.oversample:
            return 2 * self.not_ind_num
        else:
            return len(self.studies)
    
    def __getitem__(self, ind: int):
        if self.oversample:
            if ind < self.not_ind_num:  # pick not-indeterminate
                study_id = self.df_not_ind.iloc[ind].StudyInstanceUID
            else:  # pick indeterminte (oversmapled)
                study_id = self.df_ind.iloc[ind % self.ind_num].StudyInstanceUID
        else:
            study_id = self.studies[ind]
        return self.__getitem__one(study_id)

    def __getitem__one(self, study_id):
        df = self.df[self.df.StudyInstanceUID == study_id]
        # label
        label = rawlabel_to_label_study_level(df.iloc[0])
        # 3d image
        images = get_img_jpg256_all(df)
        images = np.array([img[:,:,1] for img in images])
        images = images.astype(np.float32) / 255.  # pick PE window
        # trans
        img_3d = self.transform(images)
        img_3d = np.expand_dims(np.array(img_3d), axis=0)  # (1,D,H,W)
        # import pdb; pdb.set_trace()
        return img_3d, label, study_id


def get_img_jpg256(r):
    folder = DATADIR / "train-jpegs" / r["StudyInstanceUID"] / r["SeriesInstanceUID"]
    img_path = folder / ('{:04}_'.format(r["img_prefix"]) + r["SOPInstanceUID"] + ".jpg")
    return cv2.imread(str(img_path))

def get_transform_v1():
    return alb.Compose([
        alb.RandomCrop(224, 244, p=1)
    ])
def get_transform_valid_v1():
    return alb.Compose([alb.CenterCrop(224, 244, p=1)])

def get_img_jpg256_all(df):
    imgs = []
    for index, r in df.iterrows():
        folder = DATADIR / "train-jpegs" / r["StudyInstanceUID"] / r["SeriesInstanceUID"]
        img_path = folder / ('{:04}_'.format(r["img_prefix"]) + r["SOPInstanceUID"] + ".jpg")
        imgs.append(cv2.imread(str(img_path)))
    return imgs


class RsnaDatasetTest(data.Dataset):
    """Test Time Dataset. for now, image level dataset"""
    def __init__(self, df=None):
        """df: one sop only"""
        self.df_all = df if df  else pd.read_csv(DATADIR / "test.csv")
        self.transform = get_transform_valid_v1()
        
    def __len__(self):
        return len(self.df)
    
    def set_StudyInstanceUID(self, study_id):
        self.df = self.df_all[self.df_all.StudyInstanceUID == study_id]
        assert len(self.df) > 0
        self.images, self.sop_arr = get_sorted_hu(self.df)

    def __getitem__(self, idx: int):
        # image
        image = self.images[idx]
        image = hu_to_3wins(image)
        image = self.transform(image=image)["image"]
        image = (image.astype(np.float32) / 255).transpose(2,0,1)

        return image, {}, self.sop_arr[idx]  # image, dummy_label_dict, id


class RsnaDatasetTest2(data.Dataset):
    """Test Time Dataset. for now, image level dataset"""
    def __init__(self, df=None):
        """df: one sop only"""
        self.df_all = df if df  else pd.read_csv(DATADIR / "test.csv")
        self.studies = self.df_all.StudyInstanceUID.unique()
        self.transform = get_transform_valid_v1()

    def __len__(self):
        return len(self.studies)
    
    def _trans(self, img):
        return ((self.transform(image=hu_to_3wins(img))["image"]).astype(np.float32) / 255.).transpose(2,0,1)

    def __getitem__(self, idx: int):
        study_id = self.studies[idx]
        df = self.df_all[self.df_all.StudyInstanceUID == study_id]
        images, sop_arr = get_sorted_hu(df)
        images = [self._trans(img) for img in images]
        images = np.array(images)

        return images, study_id, sop_arr


def hu_to_3wins(image):
    # 'jpg256' dataset is convert by this function
    # Windows from https://pubs.rsna.org/doi/pdf/10.1148/rg.245045008
    MAX_LENGTH = 256.
    image_lung = np.expand_dims(hu_to_windows(image, WL=-600, WW=1500), axis=-1)
    image_mediastinal = np.expand_dims(hu_to_windows(image, WL=40, WW=400), axis=-1)
    image_pe_specific = np.expand_dims(hu_to_windows(image, WL=100, WW=700), axis=-1)
    image = np.concatenate([image_mediastinal, image_pe_specific, image_lung], axis=-1)
    # import pdb; pdb.set_trace()
    rat = MAX_LENGTH / np.max(image.shape[1:])
    image = zoom(image, [rat,rat,1.], prefilter=False, order=1)
    # image = zoom(image, [1.,rat,rat,1.], prefilter=False, order=1)
    return image

def hu_to_windows(img, WL=50, WW=350):
    upper, lower = WL+WW//2, WL-WW//2
    X = np.clip(img.copy(), lower, upper)
    X = X - np.min(X)
    X = X / np.max(X)
    X = (X*255.0).astype('uint8')
    return X

def get_sorted_hu(df, folder='test'):
    d = '../input/rsna-str-pulmonary-embolism-detection/' + folder + '/' + df.StudyInstanceUID + '/' + df.SeriesInstanceUID + '/'
    dicom_files = list((d + df.SOPInstanceUID + '.dcm').unique())
    hu_images, sop_arr = load_dicom_array(dicom_files)
    return hu_images, sop_arr

def load_dicom_array(dicom_files):
    """z pos sorted dicom images and files"""
    # dicom_files = glob.glob(os.path.join(series_dir, '*.dcm'))  # series_dir to dicom_files
    dicoms = [pydicom.dcmread(d) for d in dicom_files]
    M = float(dicoms[0].RescaleSlope)
    B = float(dicoms[0].RescaleIntercept)
    # Assume all images are axial
    z_pos = [float(d.ImagePositionPatient[-1]) for d in dicoms]

    ### read with error check
    dicoms_arr = []
    for d in dicoms:
        try:
            img = d.pixel_array
        except:
            print('image error ', d)
            img = np.zeros(shape=(512,512))
        dicoms_arr.append(img)
    dicoms = np.array(dicoms_arr)
    ### read without error check
    # dicoms = np.asarray([d.pixel_array for d in dicoms])

    dicoms = dicoms[np.argsort(z_pos)]
    dicoms = dicoms * M
    dicoms = dicoms + B
    # sorted_dicom_files = np.asarray(dicom_files)[np.argsort(z_pos)]
    sorted_sop = np.asarray([os.path.basename(f)[:-4] for f in dicom_files])[np.argsort(z_pos)]
    return dicoms, sorted_sop


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


# test
if __name__ == "__main__":
    outpath = DATADIR / "split.csv"
    split_stratified(outpath)