import torch
from torch.utils.data import DataLoader

from extractors.VisualExtractor import VisualExtractor
from utils import utils


def main():
    utils.seed_everything()
    VE = VisualExtractor(model='vgg', layers=2)
    tree = utils.crawl_directory(
        '/media/theo/Hard Disk 2/projects_git/deep_video_extraction/data')
    frames, fps = utils.analyze_video(tree[0])
    feature_vector = (VE.extract(frames))


if __name__ == "__main__":
    main()
