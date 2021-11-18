import sys
from gc import collect as gc_collect
import os
from cv2 import _OutputArray_DEPTH_MASK_FLT
import numpy as np
from torch.cuda import empty_cache
from torch.utils.data import DataLoader
from tqdm import tqdm

from extractors.VisualExtractor import VisualExtractor
from utils import utils
from utils.dataset import VideoDataset

VIDEO_PATH = '/media/theo/Hard Disk 2/projects_git/deep_video_extraction/Video_smaller'
OUTPUT = 'output'

# @utils.timeit


def extractVisual(directory: str, model: str, layers: int, output: str = 'output', save: bool = True) -> None:
    tree = utils.crawl_directory(directory)
    destination = None
    predictions = []
    visual_extractor = VisualExtractor(model=model, layers=layers)
    for filepath in tree:
        print(f'Processing {filepath}')
        dataset = VideoDataset(filepath)
        dataloader = DataLoader(dataset, batch_size=16,
                                shuffle=False, num_workers=2)

        filename = filepath.split(os.sep)[-1].split('.')[0]
        classname = filepath.split(os.sep)[-2]
        destination = os.path.join(output, classname)
        if not utils.is_dir(destination):
            utils.create_dir(destination)
        predictions = visual_extractor.extract(dataloader)

        if (save):
            np.save(os.path.join(destination, f'{filename}.npy'), predictions)

        del predictions
        empty_cache()
        gc_collect()


def main():
    OUTPUT = 'output'
    if not utils.is_dir(OUTPUT):
        utils.create_dir(OUTPUT)
    else:
        OUTPUT = OUTPUT + '_' + utils.get_timestamp()
        print(f'Creating new with name: {OUTPUT}')
        utils.create_dir(OUTPUT)

    extractVisual(directory=VIDEO_PATH, model='vgg',
                  layers=2, output=OUTPUT, save=True)


if __name__ == "__main__":
    main()
