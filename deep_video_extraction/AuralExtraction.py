import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

vgg19 = models.vgg19(pretrained=True)


class AuralExtraction(nn.Module):
    def __init__(self):
        super(AuralExtraction).__init__()

    def forward(self, x):
        return x
