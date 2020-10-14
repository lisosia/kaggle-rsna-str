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

class ImgLossPE(nn.Module):
    """Img Level Predction Loss"""
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()
    def forward(self, input, target):
        total_loss = \
            1./1 * self.loss(input["pe_present_on_image"], target["pe_present_on_image"].float()) + \
            1./4 * self.loss(input["rightsided_pe"], target["rightsided_pe"].float()) + \
            1./4 * self.loss(input["leftsided_pe"], target["leftsided_pe"].float()) + \
            1./2 * self.loss(input["central_pe"], target["central_pe"].float())
        return total_loss


class ImgLossChronic(nn.Module):
    """Img Level Predction Loss"""
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()
    def forward(self, input, target):
        total_loss = \
            1./2 * self.loss(input["pe_present_on_image"], target["pe_present_on_image"].float()) + \
            1./2 * self.loss(input["chronic_pe"], target["chronic_pe"].float()) + \
            1./2 * self.loss(input["acute_and_chronic_pe"], target["acute_and_chronic_pe"].float())
            # 1./4 * self.loss(input["acute_pe"], target["acute_pe"].float())  # acute_pe is dominant. so most diddicult (you must predict "1" most freqently)
        return total_loss


class ImgLossRL(nn.Module):
    """Img Level Predction Loss"""
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()
    def forward(self, input, target):
        # pdb.set_trace()
        return self.loss(input["rv_lv_ratio_gte_1"], target["rv_lv_ratio_gte_1"].float())


# Indeterminate loss (indeterminate+qa_constrast+qa_motion)
class ImgLossInd(nn.Module):
    """Img Level Predction Loss"""
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()
    def forward(self, input, target):
        total_loss = \
            1./2 * self.loss(input["indeterminate"], target["indeterminate"].float()) + \
            1./4 * self.loss(input["qa_contrast"], target["qa_contrast"].float()) + \
            1./4 * self.loss(input["qa_motion"], target["qa_motion"].float())
        return total_loss

