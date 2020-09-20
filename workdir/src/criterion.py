import torch
import torch.nn as nn
import torch.nn.functional as F


class ImgLoss(nn.Module):
    """Img Level Predction Loss"""
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss
    def forward(self, input, target)