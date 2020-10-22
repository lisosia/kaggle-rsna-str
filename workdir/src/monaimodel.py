import numpy as np
import torch

import monai
from monai.transforms import LoadNifti, Randomizable, apply_transform
from monai.transforms import AddChannel, Compose, RandRotate90, Resize, ScaleIntensity, ToTensor
from monai.utils import get_seed

monai_model_file = "output_jan/monai3d_160_3ch_1e-5_20ep_aug_6targets_0_0.26139065623283386.pth"
print("use monai model:", monai_model_file)

target_cols = [
        'rv_lv_ratio_gte_1', # exam level
        "central_pe",
        "leftsided_pe",
        "rightsided_pe",
        "acute_and_chronic_pe",
        "chronic_pe"
    ]
out_dim = len(target_cols)
image_size = 100

val_transforms = Compose([ScaleIntensity(), Resize((image_size, image_size, image_size)), ToTensor()])
val_transforms.set_random_state(seed=42)

monai_model = monai.networks.nets.densenet.densenet121(spatial_dims=3, in_channels=3, out_channels=out_dim).to("cuda")
monai_model.load_state_dict(torch.load(monai_model_file), strict=True)
monai_model.eval()  

def pred_monai(imgs512):
    imgs = imgs512[:, :, 43:-55, 43:-55]
    img_monai = imgs[int(imgs.shape[0] * 0.25 ): int(imgs.shape[0] * 0.75)]
    img_monai = np.transpose(img_monai, (1,2,3,0))
    img_monai = apply_transform(val_transforms, img_monai)
    img_monai = np.expand_dims(img_monai, axis=0)
    img_monai = torch.from_numpy(img_monai).cuda()
    monai_preds = torch.sigmoid(monai_model(img_monai)).cpu().detach().numpy().squeeze()

    return monai_preds
