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


# Indeterminate loss (indeterminate+qa_constrast+qa_motion)
class ImgLossInd(nn.Module):
    """Img Level Predction Loss"""
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()
    def forward(self, input, target):
        total_loss = \
            2 * self.loss(input["indeterminate"], target["indeterminate"].float()) + \
            1 * self.loss(input["qa_contrast"], target["qa_contrast"].float()) + \
            1 * self.loss(input["qa_motion"], target["qa_motion"].float())
        return total_loss

