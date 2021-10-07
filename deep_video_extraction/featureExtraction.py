from utils import utils
from extractors.VisualExtractor import VisualExtractor
from torch.utils.data import DataLoader
import torch


def main():
    device = utils.device()
    utils.seed_everything()
    VE = VisualExtractor(4)
    print(VE)
    tree = utils.crawl_directory(
        '/media/theo/Hard Disk 2/projects_git/deep_video_extraction/data')
    frames = utils.analyze_video(tree[0])
    print(utils.device())
    print(len(frames))


if __name__ == "__main__":
    main()
