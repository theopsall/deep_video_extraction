import gc
import os

import torch
from torch.utils.data import DataLoader, dataloader
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

from config import MEAN, STD
from extractors.VisualExtractor import VisualExtractor
from utils import utils
from utils.dataset import VideoDataset
from time import sleep
import time
import sys

VIDEO_PATH = '/home/theo/Documents/deep_video_extraction/visual_features'


@utils.timeit
def extractVisual(directory: str, model: str, layers: int):
    tree = utils.crawl_directory(directory)
    predictions = []
    visual_extractor = VisualExtractor(model='vgg', layers=2)
    for filename in tree:
        print(f'Processing {filename}')
        dataset = VideoDataset(filename)
        dataloader = DataLoader(dataset, batch_size=32,
                                shuffle=False, num_workers=4)
        predictions.append(visual_extractor.extract(dataloader))
    return predictions


def main():
    pass


if __name__ == "__main__":
    main()
