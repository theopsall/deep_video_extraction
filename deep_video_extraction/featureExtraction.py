import os
from gc import collect as gc_collect

import numpy as np
from cv2 import _OutputArray_DEPTH_MASK_FLT
from extractors.VisualExtractor import VisualExtractor
from torch.cuda import empty_cache
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import utils
from utils.dataset import SpectrogramDataset, VideoDataset


def extract_features(
    directory: str,
    model: str,
    layers: int ,
    flatten: bool,
    output: str,
    save: bool,
    dataset_class,
) -> None:
    tree = utils.crawl_directory(directory)
    destination = None
    predictions = []

    visual_extractor = VisualExtractor(model=model, layers=layers, flatten=flatten)
    pbar = tqdm(tree, desc=f"Processing ")
    for filepath in pbar:
        
        # Extract filename and classname
        filename = os.path.splitext(filepath.split(os.sep)[-1])[0]
        classname = filepath.split(os.sep)[-2]
        destination = os.path.join(output, classname)
        pbar.set_description(f"Processing {filename}")
        
        if not utils.is_dir(destination):
            utils.create_dir(destination)

        # Load and process the dataset
        dataset = dataset_class(filepath)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)

        # Perform feature extraction
        predictions = visual_extractor.extract(dataloader)
        if save:
            np.save(os.path.join(destination, f"{filename}.npy"), predictions)

        del predictions
        empty_cache()
        gc_collect()


def extract_visual_features(
    directory: str,
    model: str ,
    layers: int ,
    flatten: bool = False,
    output: str = "deep_visual_features",
    save: bool = True,
) -> None:
    extract_features(
        directory=directory,
        model=model,
        layers=layers,
        flatten=flatten,
        output=output,
        save=save,
        dataset_class=VideoDataset,
    )


def extract_aural_features(
    directory: str,
    model: str,
    layers: int ,
    flatten: bool = False,
    output: str = "deep_visual_features",
    save: bool = True,
) -> None:
    extract_features(
        directory=directory,
        model=model,
        layers=layers,
        flatten=flatten,
        output=output,
        save=save,
        dataset_class=SpectrogramDataset,
    )


def audio_extraction(
    directory: str, output: str = "aural_output", save: bool = True
) -> None:
    tree = utils.crawl_directory(directory)
    destination = None
    predictions = []
    for filepath in tqdm(tree, desc="Processing files"):
        # Print the current file being processed
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
    for filepath in tqdm(tree, desc="Processing files"):
        # Print the current file being processed
        print(f"Getting Spectrogram {filepath}")

        filename = os.path.splitext(filepath.split(os.sep)[-1])[0]
        classname = filepath.split(os.sep)[-2]
        destination = os.path.join(output, classname)
        if not utils.is_dir(destination):
            utils.create_dir(destination)
        utils.get_spectrogram(filepath, os.path.join(destination, f"{filename}.png"))
        gc_collect()
