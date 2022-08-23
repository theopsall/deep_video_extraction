from gc import collect as gc_collect

import numpy as np
import torch
import torch.nn as nn
from config import MEAN, STD
from numpy import ndarray
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm
from utils.utils import clean_GPU, device


class VisualExtractor(nn.Module):

    def __init__(
        self, model: str = "vgg", layers: int = 1, flatten: bool = False
    ) -> None:
        super(VisualExtractor, self).__init__()
        self.name = model
        self._model = self.get_model()
        self.layers = abs(layers)
        self.flatten = flatten
        self.model = nn.Sequential(*list(self._model.children())[: -self.layers])
        self.device = device()
        self.model.to(self.device)
        self.model.eval()
        self.normalize = transforms.Normalize(mean=MEAN, std=STD)
        self.to_tensor = transforms.ToTensor()

    def get_model(self):
        if self.name == "resnet":
            return models.resnet18(pretrained=True)
        if self.name == "vgg":
            return models.vgg19(pretrained=True)
        if self.name == "alexnet":
            return models.alexnet(pretrained=True)
        raise Exception(f"Wrong model selection: {self.name}")

    def transform(self, x):
        return self.normalize(self.to_tensor(x)).unsqueeze(0).to(self.device)

    def extract(self, testLoader: DataLoader) -> ndarray:
        out = []
        with torch.no_grad():
            for batch in testLoader:
                batch = batch.to(self.device)
                output = self.model(batch).to("cpu")
                [
                    out.append(np.array(t).flatten() if self.flatten else np.array(t))
                    for t in output
                ]
                del output
                del batch
                gc_collect()
                torch.cuda.empty_cache()
        return np.array(out)
