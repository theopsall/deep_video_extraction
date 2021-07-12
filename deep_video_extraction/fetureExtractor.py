import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

vgg19 = models.vgg19(pretrained=True)


class FeatureExtraction(nn.Module):
    def __init__(self):
        super(FeatureExtraction).__init__()

    def forward(self, x):
        return x
