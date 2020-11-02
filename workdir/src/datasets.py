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
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

DATADIR = Path("../input/rsna-str-pulmonary-embolism-detection/")

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

        "pe_present_portion": row["pe_present_portion"],  # used for pe_present metric
    })
    # if multi position has pe, each slice may have one individual pe (left or right or center)
    # so use soft-label.
    if ret["pe_present_on_image"] > 0.5 and (ret["rightsided_pe"]+ret["leftsided_pe"]+ret["central_pe"]) > 1:
        EPS=0.01  # note: 85% of pos_exam is rightsided
        ret["rightsided_pe"] = np.clip(ret["rightsided_pe"], 0, 1-EPS)
        ret["leftsided_pe"] = np.clip(ret["leftsided_pe"], 0, 1-EPS)
        ret["central_pe"] = np.clip(ret["central_pe"], 0, 1-EPS)
    if False:
        EPS = 1e-2
        ret["pe_present_on_image"] = np.clip(ret["pe_present_on_image"], EPS, 1-EPS)
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
    def __init__(self, fold, phase, oversample=None, cutmix_prob=0.0,\
        train_transform_str='get_transform_v2_512', val_transform_str='get_transform_valid_v1_512'):
        """
        oversample:
            example num is... pe_present_portion(all)/pe_present_portion(pos_exam) ~= 3.3
            the metrics weights more for slices for high portion exam. so a little more big ratio may be better
        """
        assert phase in ["train", "valid"]
        self.oversample = oversample if (oversample and phase=="train") else None
        self.cutmix_prob = cutmix_prob
        self.phase = phase
        self.datafir = DATADIR
        # prepare df
        df = pd.read_csv(DATADIR / "train.csv")
        df_fold = pd.read_csv(DATADIR / "split.csv")
        df_fold = df_fold.merge(pd.read_csv(DATADIR / "study_pos_portion.csv"), on="StudyInstanceUID")  # add pe_present_portion col
        df = df.merge(df_fold, on="StudyInstanceUID")  # fold row
        df_prefix = pd.read_csv(DATADIR / "sop_to_prefix.csv")
        df = df.merge(df_prefix, on="SOPInstanceUID")  # img_prefix row
        if phase == "train":
            self.df = df[df.fold != fold]
            if train_transform_str == 'get_transform_v2_512':
                self.transform = get_transform_v2_512()
            if train_transform_str == 'get_transform_v4_512':
                self.transform = get_transform_v4_512()

            print("train dataset transform", self.transform)
        elif phase == "valid":
            self.df = df[df.fold == fold]
            if val_transform_str == 'get_transform_valid_v1_512':
                self.transform = get_transform_valid_v1_512()

        # self.df = self.df.iloc[:600]  # debug
        if self.oversample:
            assert isinstance(self.oversample, int) and self.oversample > 1  # sample positive `oversample` times
            self.pe_present_num = len(self.df[self.df.pe_present_on_image == 1])
            print(f"oversample: pe_present/total={self.pe_present_num}/{len(self.df)} . oversample={self.oversample}")

    def __len__(self):
        if self.oversample:
            return len(self.df) + (self.oversample - 1) * self.pe_present_num
        else:
            return len(self.df)

    def __getitem__(self, idx: int):
        if self.phase == "train" and np.random.rand() < self.cutmix_prob:
            img1, y1,  sop1 = self.__getitem__wrap1(idx)
            img2, y2, _sop2 = self.__getitem__wrap1(np.random.randint(self.__len__()))
            new_img, new_y = transforms.cutmix(img1, img2, y1, y2)
            return new_img, new_y, sop1
        else:
            return self.__getitem__wrap1(idx)

    def __getitem__wrap1(self, idx: int):
        if self.oversample:
            if idx < len(self.df):
                new_idx = idx
            else:
                new_idx = (idx - len(self.df)) % self.pe_present_num
            return self.__getitem__one(new_idx)
        else:
            return self.__getitem__one(idx)

    def __getitem__one(self, idx: int):
        sample = self.df.iloc[idx, :]
        # label
        label = rawlabel_to_label(sample)
        # image
        # image = get_img_jpg256(sample)
        image = get_img_jpg512(sample)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        image = (image.astype(np.float32) / 255).transpose(2,0,1)

        if 'weight' in sample:
            weight = sample.weight
        else:
            weight = 1 # dfにカラムがない場合

        return image, label, sample.SOPInstanceUID, weight


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

def get_img_jpg512(r):
    folder = DATADIR / "train-jpegs-512"
    img_path = folder / (r["SOPInstanceUID"] + ".jpg")
    return cv2.imread(str(img_path))[:,:,::-1]

def get_transform_v1():
    return alb.Compose([
        alb.RandomCrop(224, 244, p=1)
    ])
def get_transform_valid_v1():
    return alb.Compose([alb.CenterCrop(224, 244, p=1)])

def get_transform_valid_v1_512():
    return alb.Compose([alb.CenterCrop(448, 448, p=1)])

def get_transform_valid_another_crop_512():
   return alb.Compose([alb.Crop(12, 42, 500, 460, p=1.0)])

def get_transform_v2():
    return alb.Compose([
        alb.RandomCrop(224, 244, p=1),
        alb.HorizontalFlip(p=0.5),
        alb.VerticalFlip(p=0.5),
    ])

def get_transform_v2_512():
    return alb.Compose([
        alb.RandomCrop(448, 448, p=1),
        alb.HorizontalFlip(p=0.5),
        alb.VerticalFlip(p=0.5),
    ])

def get_transform_v3_512():
    return alb.Compose([
        alb.RandomCrop(448, 448, p=1),
        alb.HorizontalFlip(p=0.5),
        alb.VerticalFlip(p=0.5),
    ])

def get_transform_v4_512():
    return alb.Compose([
        alb.RandomCrop(448, 448, p=1),
        alb.VerticalFlip(p=0.5),
    ])

def get_transform_another_crop_512():
    return alb.Compose([
        alb.Crop(12, 42, 500, 460, p=1.0),
        alb.HorizontalFlip(p=0.5),
        alb.VerticalFlip(p=0.5),
    ])

def get_transform_v2_512_exclude_crop():
    return alb.Compose([
        alb.HorizontalFlip(p=0.5),
        alb.VerticalFlip(p=0.5),
    ])

def get_transform_v3():
    return alb.Compose([
        alb.ShiftScaleRotate(shift_limit=0, scale_limit=0.1, rotate_limit=20, interpolation=cv2.INTER_AREA,
                             border_mode=cv2.BORDER_CONSTANT, value=0, p=0.8),
        alb.RandomCrop(224, 244, p=1),
        alb.HorizontalFlip(p=0.5),
        alb.VerticalFlip(p=0.5),
        alb.CoarseDropout(min_holes=4, max_holes=7, min_height=12, min_width=12, max_height=30, max_width=30, p=0.85)
    ])

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
        self.df_all = df if df is not None  else pd.read_csv(DATADIR / "test.csv")
        self.studies = self.df_all.StudyInstanceUID.unique()
        self.transform = get_transform_valid_v1_512()

    def __len__(self):
        return len(self.studies)
    
    def _trans(self, img):
        return ((self.transform(image=hu_to_3wins_fast_512(img))["image"]).astype(np.float32) / 255.).transpose(2,0,1)

    def __getitem__(self, idx: int):
        study_id = self.studies[idx]
        df = self.df_all[self.df_all.StudyInstanceUID == study_id]
        images, sop_arr, z_pos_arr = get_sorted_hu(df)
        images = [self._trans(img) for img in images]
        images = np.array(images)

        return images, study_id, sop_arr, z_pos_arr


class RsnaDatasetTest3(data.Dataset):
    """Image level, All images"""
    def __init__(self):
        """df: test.csv"""
        self.df = pd.read_csv(DATADIR / "test.csv")
        self.dir = '../input/rsna-str-pulmonary-embolism-detection/test/'
        self.transform = get_transform_valid_v1_512()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        sample = self.df.iloc[idx]
        image, z_pos = load_dicom(self.dir + '/' + sample.StudyInstanceUID + '/' + sample.SeriesInstanceUID + '/' + sample.SOPInstanceUID + '.dcm')
        image = hu_to_3wins_fast_512(image)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        image = (image.astype(np.float32) / 255).transpose(2,0,1)

        return image, sample.StudyInstanceUID, sample.SOPInstanceUID, z_pos  # image, dummy_label_dict, id

class RsnaDatasetTest3Valid(data.Dataset):
    """Image level, All images, for local validation"""
    def __init__(self, df):
        """df: test.csv"""
        self.df = df
        self.transform = get_transform_valid_v1_512()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        sample = self.df.iloc[idx]
        image = get_img_jpg512(sample)
        image = self.transform(image=image)["image"]
        image = (image.astype(np.float32) / 255).transpose(2,0,1)

        return image, sample.StudyInstanceUID, sample.SOPInstanceUID  # image, dummy_label_dict, id


# def hu_to_windows_fast(img, WL=50, WW=350):  # Causion! Slightly Difference Behaior
#     upper, lower = WL+WW//2, WL-WW//2
#     X = np.clip(img, lower, upper)
#     X = (X - lower) / float(upper - lower)
#     return X * 255.0
def hu_to_3wins_fast_512(image):
    # MAX_LENGTH = 512
    image_lung        = np.expand_dims(hu_to_windows(image, WL=-600, WW=1500), axis=-1)
    image_mediastinal = np.expand_dims(hu_to_windows(image, WL=40, WW=400), axis=-1)
    image_pe_specific = np.expand_dims(hu_to_windows(image, WL=100, WW=700), axis=-1)
    image = np.concatenate([image_mediastinal, image_pe_specific, image_lung], axis=-1)
    # # assert image.shape == (512, 512, 3)  # all ok for test/
    # # _image_saved = image.copy()
    # rat = MAX_LENGTH / np.max(image.shape[1:])
    # image = zoom(image, [rat,rat,1.], prefilter=False, order=1)
    # # assert ( _image_saved == image ).all()  # all ok for test/
    return image



def hu_to_3wins(image, MAX_LENGTH=256.):
    # 'jpg256' dataset is convert by this function
    # Windows from https://pubs.rsna.org/doi/pdf/10.1148/rg.245045008
    
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
    hu_images, sop_arr, z_pos_arr = load_dicom_array(dicom_files)
    return hu_images, sop_arr, z_pos_arr


# def load_dicom(dicom_file_path):
#     """return img (meda_data_dict)"""
#     d = pydicom.dcmread(dicom_file_path)
#     M = float(d.RescaleSlope)
#     B = float(d.RescaleIntercept)
#     try:
#         img = d.pixel_array
#     except:
#         print('image error ', d)
#         img = np.zeros(shape=(512,512))
#     img = img * M
#     img = img + B

#     z_pos = float(d.ImagePositionPatient[-1])

#     return img, z_pos
#     # return img, dict( [(e.keyword, e.value) for e in d.iterall()] )  # not used now. ignore

print("========== load_dicom_array() NOT USE TRY-EXCEPT ======================")
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
        img = d.pixel_array
        # try:
        #     img = d.pixel_array
        # except:
        #     print('image error ', d)
        #     img = np.zeros(shape=(512,512))
        dicoms_arr.append(img)
    dicoms = np.array(dicoms_arr)
    ### read without error check
    # dicoms = np.asarray([d.pixel_array for d in dicoms])

    dicoms = dicoms[np.argsort(z_pos)]
    dicoms = dicoms * M
    dicoms = dicoms + B
    # sorted_dicom_files = np.asarray(dicom_files)[np.argsort(z_pos)]
    sorted_sop = np.asarray([os.path.basename(f)[:-4] for f in dicom_files])[np.argsort(z_pos)]
    return dicoms, sorted_sop, np.array(z_pos)[np.argsort(z_pos)]


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
