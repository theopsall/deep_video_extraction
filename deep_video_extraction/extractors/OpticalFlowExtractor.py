from gc import collect as gc_collect

import numpy as np
import torch
import torch.nn as nn
from numpy import ndarray
from torchvision import models, transforms
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from tqdm import tqdm
from utils.utils import clean_GPU, device


class OpticalFlowExtractor(nn.Module):

    def __init__(
        self, model: str = "vgg", layers: int = 1, flatten: bool = False
    ) -> None:
        super(OpticalFlowExtractor, self).__init__()
        self.name = model
        self.weights = Raft_Large_Weights.DEFAULT
        self._model = raft_large(weights=self.weights, progress=False)
        self.layers = abs(layers)
        self.flatten = flatten
        self.model = nn.Sequential(*list(self._model.children())[: -self.layers])
        self.device = device()
        self.model.to(self.device)
        self.model.eval()


    def extract(self, img_1,img_2) -> ndarray:
        out = []
        with torch.no_grad():
            img_1 = img_1.to(self.device)
            img_2 = img_2.to(self.device)
            output = self.model(img_1,img_2).to("cpu")
            [out.append(np.array(t).flatten() if self.flatten else np.array(t))for t in output]
            del img_1
            del img_2
            gc_collect()
            torch.cuda.empty_cache()
        return np.array(out)
