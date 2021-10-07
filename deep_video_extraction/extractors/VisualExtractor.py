import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets


class VisualExtractor(nn.Module):
    def __init__(self, layers: int = 1) -> None:
        super(VisualExtractor, self).__init__()
        self._rn18 = models.resnet18(pretrained=True)
        self.layers = abs(layers)
        self.model = nn.Sequential(*list(self._rn18.children())[:-self.layers])
        print(self.model)

    def forward(self, x):
        return self.model(x)

    def predict(self, testLoader) -> list:
        pass
