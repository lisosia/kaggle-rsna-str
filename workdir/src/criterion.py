import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

class ImgLoss(nn.Module):
    """Img Level Predction Loss"""
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()
    def forward(self, input, target):
        # pdb.set_trace()
        return self.loss(input["pe_present_on_image"], target["pe_present_on_image"].float())
