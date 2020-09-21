import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

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

# img level model
def get_img_model(config: dict):
    return ImgModel(archi="efficientnet_b0")
