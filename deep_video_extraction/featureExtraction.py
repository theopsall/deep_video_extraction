import os
import sys
from gc import collect as gc_collect

import numpy as np
from cv2 import _OutputArray_DEPTH_MASK_FLT
from torch.cuda import empty_cache
from torch.utils.data import DataLoader

from extractors.VisualExtractor import VisualExtractor
from utils import utils
from utils.dataset import VideoDataset


def extract_visual(
    directory: str,
    model: str = "vgg",
    layers: int = 2,
    flatten: bool = False,
    output: str = "visual_output",
    save: bool = True,
) -> None:
    tree = utils.crawl_directory(directory)
    destination = None
    predictions = []

    visual_extractor = VisualExtractor(model=model, layers=layers, flatten=flatten)
    for filepath in tree:
        print(f"Processing {filepath}")
        dataset = VideoDataset(filepath)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)

        filename = os.path.splitext(filepath.split(os.sep)[-1])[0]
        classname = filepath.split(os.sep)[-2]
        destination = os.path.join(output, classname)
        if not utils.is_dir(destination):
            utils.create_dir(destination)
        predictions = visual_extractor.extract(dataloader)

        if save:
            np.save(os.path.join(destination, f"{filename}.npy"), predictions)

        del predictions
        empty_cache()
        gc_collect()


def audio_extraction(
    directory: str, output: str = "aural_output", save: bool = True
) -> None:
    tree = utils.crawl_directory(directory)
    destination = None
    predictions = []
    for filepath in tree:
        print(f"Processing {filepath}")

        filename = os.path.splitext(filepath.split(os.sep)[-1])[0]
        classname = filepath.split(os.sep)[-2]
        destination = os.path.join(output, classname)
        if not utils.is_dir(destination):
            utils.create_dir(destination)
        utils.sound_isolation(filepath, os.path.join(destination, f"{filename}.wav"))


def extract_spectros(
    directory: str, output: str = "spectrograms_output", save: bool = True
) -> None:
    tree = utils.crawl_directory(directory)
    destination = None
    predictions = []
    for filepath in tree:
        print(f"Getting Spectrogram  {filepath}")
        filename = os.path.splitext(filepath.split(os.sep)[-1])[0]
        classname = filepath.split(os.sep)[-2]
        destination = os.path.join(output, classname)
        if not utils.is_dir(destination):
            utils.create_dir(destination)
        utils.get_spectrogram(filepath, os.path.join(destination, f"{filename}.png"))
        gc_collect()
