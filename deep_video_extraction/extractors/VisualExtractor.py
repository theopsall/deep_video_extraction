import torch
import torch.nn as nn
from config import MEAN, STD
from torch.utils.data import DataLoader
from torchvision import models, transforms
from utils.utils import device
from numpy import ndarray


class VisualExtractor(nn.Module):
    def __init__(self, model: str = 'resnet', layers: int = 1) -> None:
        super(VisualExtractor, self).__init__()
        self.name = model
        self._model = self.get_model()
        self.layers = abs(layers)
        self.model = nn.Sequential(
            *list(self._model.children())[:-self.layers])
        self.device = device()
        self.model.to(self.device)
        self.model.eval()
        self.normalize = transforms.Normalize(mean=MEAN,
                                              std=STD)
        self.to_tensor = transforms.ToTensor()

    def get_model(self):
        if(self.name == 'resnet'):
            return models.resnet18(pretrained=True)
        if(self.name == 'vgg'):
            return models.vgg19(pretrained=True)
        if(self.name == 'alexnet'):
            return models.alexnet(pretrained=True)
        raise Exception(f'Wrong model selection: {self.name}')

    def transform(self, x):
        return self.normalize(self.to_tensor(
            x)).unsqueeze(0).to(self.device)

    def forward(self, x):
        x = self.transform(x)
        out = self.model(x)
        return out

    def extract(self, x):
        if type(x) == ndarray:
            with torch.no_grad():
                x = self.transform(x)
                out = self.model(x)
                print(out.shape)
                out = torch.flatten(out)
                print(out.shape)
                return out
        if type(x) == list:
            out = []
            with torch.no_grad():
                for frame in x:
                    frame = self.transform(frame)
                    output = self.model(frame)
                    output = torch.flatten(output)
                    out.append(output)

            return out
