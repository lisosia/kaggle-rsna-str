from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pandas as pd
import numpy as np
import glob
import pydicom
import cv2
import os, os.path as osp
from PIL import Image
import torch

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def load_dicom_array(f):
    dicom_files = glob.glob(osp.join(f, '*.dcm'))
    dicoms = [pydicom.dcmread(d) for d in dicom_files]
    M = float(dicoms[0].RescaleSlope)
    B = float(dicoms[0].RescaleIntercept)
    # Assume all images are axial
    z_pos = [float(d.ImagePositionPatient[-1]) for d in dicoms]
    dicoms = torch.tensor(np.array([d.pixel_array for d in dicoms]).astype('float'))
    dicoms = dicoms[np.argsort(z_pos)]
    dicoms = dicoms * M
    dicoms = dicoms + B
    return dicoms, np.asarray(dicom_files)[np.argsort(z_pos)]


def save_array(X, save_dir, file_names):
    for ind, img in enumerate(X):
        savefile = osp.join(save_dir, file_names[ind])
        if not osp.exists(osp.dirname(savefile)):
            os.makedirs(osp.dirname(savefile))
        Image.fromarray(np.uint8(img)).save(str(osp.join(save_dir, file_names[ind])), quality=100)

def edit_filenames(files):
    dicoms = [f"{ind:04d}_{f.split('/')[-1].replace('dcm','png')}" for ind,f in enumerate(files)]
    series = ['/'.join(f.split('/')[-3:-1]) for f in files]
    return [osp.join(s,d) for s,d in zip(series, dicoms)]


class Lungs(Dataset):
    def __init__(self, dicom_folders):
        self.dicom_folders = dicom_folders
    def __len__(self): return len(self.dicom_folders)
    def get(self, i):
        return load_dicom_array(self.dicom_folders[i])
    def __getitem__(self, i):
        try:
            return self.get(i)
        except Exception as e:
            print(e)
            return None

SAVEDIR = 'train_imgs/'

if not osp.exists(SAVEDIR): os.makedirs(SAVEDIR)

df = pd.read_csv('../input/train.csv')
dicom_folders = list(('../input/train/' + df.StudyInstanceUID + '/'+ df.SeriesInstanceUID).unique())

dset = Lungs(dicom_folders)
loader = DataLoader(dset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x)

def window(img, WL=50, WW=350):
    upper, lower = WL+WW//2, WL-WW//2
    X = torch.clamp(img.clone(), lower, upper)
    X = X - torch.min(X)
    X = X / torch.max(X)
    X = (X*255.0).int()
    return X

for i,data in tqdm(enumerate(loader), total=len(loader)):
    data = data[0]
    if type(data) == type(None):
        print(f'error: {i}')
        continue

    image, files = data
    image_lung = torch.unsqueeze(window(image, WL=-600, WW=1500), axis=3)
    image_mediastinal = torch.unsqueeze(window(image, WL=40, WW=400), axis=3)
    image_pe_specific = torch.unsqueeze(window(image, WL=100, WW=700), axis=3)
    image = torch.cat([image_mediastinal, image_pe_specific, image_lung], axis=3)
    files = edit_filenames(files)
    save_array(np.array(image.cpu()), SAVEDIR, files)
