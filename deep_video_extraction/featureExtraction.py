import gc
import os

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

from config import MEAN, STD
from extractors.VisualExtractor import VisualExtractor
from utils import utils
from utils.dataset import VideoDataset


def main():
    VIDEO_PATH = '/media/theo/Hard Disk 2/projects_git/deep_video_extraction/Video_smaller'
    utils.seed_everything()
    video_tree = utils.crawl_directory(VIDEO_PATH)
    print(f'Total Videos in directory {len(video_tree)}')

    VE = VisualExtractor(model='vgg', layers=2)

    for video in video_tree:
        videoSet = VideoDataset(video_path=video, transform=Compose([ToTensor(),
                                                                     Normalize(mean=MEAN, std=STD)])
                                )
        print("Finished videoSet")
        loader = DataLoader(videoSet, batch_size=32, num_workers=1)
        prediction = VE.extract(loader)
        print(f'Prediction shape: {prediction.shape}')


if __name__ == "__main__":
    main()
