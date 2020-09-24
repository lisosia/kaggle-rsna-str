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


# img level model
def get_img_model(config: dict):
    return ImgModel(archi="efficientnet_b0")
