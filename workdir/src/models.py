import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np

class ImgModel(nn.Module):
    def __init__(self, archi, pretrained=True):
        super().__init__()
        self.base = timm.create_model(archi, pretrained=pretrained, num_classes=1)
    def forward(self, x):
        x = self.base(x)
        x = x.squeeze(-1)
        return {
            "pe_present_on_image": x
        }


## Indeterminate: 3ch image-level model
## 以下3点が主なTODO. acc をみつつ判断する
# - 3D方向に見ないとqa_motion等は検知できない疑いがある
# - qa_motion 特に、resizeしないほうがずっとずっとよい可能性がある
# - PE のみ, あるいは raw hu image のみでよいかもしれない (そちらのほうがoverfitリスクが小さいかも）
class ImgModelInd(nn.Module):
    def __init__(self, archi, pretrained=True):
        super().__init__()
        self.base = timm.create_model(archi, pretrained=pretrained, num_classes=3)
    def forward(self, x):
        x = self.base(x)  # [B,3] 3->indeterminate,qa_contrast,qa_motion
        return {
            "indeterminate": x[:, 0],
            "qa_contrast": x[:, 1],
            "qa_motion": x[:, 2],
        }


class ImgModelIndAverageBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Conv2d(1,1,1)
    def forward(self, x):
        b = x.size(0)
        y = 0.020484822355039723
        l = np.log(y / (1-y))
        pred = torch.from_numpy(l*np.ones(b)).cuda()
        return {
            "indeterminate": pred,
            "qa_contrast": pred,
            "qa_motion": pred,
        }


class ImgModelIndShallow(nn.Module):
    """shallow model"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, (3, 3))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, (3, 3))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, (3, 3))
        self.bn3 = nn.BatchNorm2d(128)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(128, 256)
        self.drop1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(256, 3)

    def forward(self, x):
        batchsize = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.gap(x)
        x = x.view(batchsize, -1)
        x = F.relu(self.drop1(self.fc1(x)))
        x = self.fc2(x)

        return {
            "indeterminate": x[:, 0],
            "qa_contrast": x[:, 1],
            "qa_motion": x[:, 2],
        }

import resnet3d.models.resnet
class ImgModelInd3D(nn.Module):
    """3D model for indeterminate"""
    def __init__(self, model_depth, pretrained=True):
        super().__init__()
        self.base = resnet3d.models.resnet.generate_model(
                    model_depth=model_depth, n_input_channels=1, n_classes=3)
        # # load Pretrain
        # ckpt_path = {18: "resnet3d/r3d18_KM_200ep.pth"}[model_depth]
        # ckpt =  torch.load(ckpt_path, map_location='cpu')
        # self.base.load_state_dict(ckpt['state_dict'])
        # self.base.fc = nn.Linear(self.base.fc.in_features, 3)

    def forward(self, x):
        # x = x.repeat(1,3,1,1,1)  # for pretrain
        x = self.base(x)
        return {
            "indeterminate": x[:, 0],
            "qa_contrast": x[:, 1],
            "qa_motion": x[:, 2],
        }


# img level model
def get_img_model(config: dict):
    return ImgModel(archi="efficientnet_b0")
