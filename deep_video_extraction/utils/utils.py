import argparse
import os
import random

import cv2
import numpy as np
import torch
from PIL import Image

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mkv', 'webm'}


def device():
    """
    Check if cuda is avaliable else choose the cpu
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu') # For testing purposes
    print(f'pyTorch is using {device}')
    return device


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_GPU():
    return torch.cuda.empty_cache()


def is_dir(directory: str) -> bool:
    return os.path.isdir(directory)


def is_file(filename: str) -> bool:
    return os.path.isfile(filename)


def is_video(video: str) -> bool:
    return video.split('.')[-1] in ALLOWED_EXTENSIONS


def crawl_directory(directory: str) -> list:
    if not is_dir(directory):
        raise FileNotFoundError

    subdirs = [folder[0] for folder in os.walk(directory)]
    tree = []
    for subdir in subdirs:
        files = next(os.walk(subdir))[2]
        for _file in files:
            tree.append(os.path.join(subdir, _file))
    return tree


def read_video():
    pass


def allowed_file(filename: str):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def isolate_audio(path: str):
    pass


def get_spectrogram(audio):
    pass


def analyze_video(video: str) -> np.ndarray:
    cap = cv2.VideoCapture(video)
    try:
        # frame per second for the current video in order to average the frames
        fps = int(cap.get(cv2.CAP_PROP_FPS))
    except ValueError:
        assert f"Cannot convert video {video} fps to integer"
    print(fps)
    success = True
    batches = []

    while success:
        success, frame = cap.read()
        if success:
            batches.append(np.array(frame))
    return batches, fps


def analyze_video_in_batches(video: str, batch_size: int = 32):
    cap = cv2.VideoCapture(video)
    try:
        # frame per second for the current video in order to average the frames
        fps = int(cap.get(cv2.CAP_PROP_FPS))
    except ValueError:
        assert f"Cannot convert video {video} fps to integer"
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    except ValueError:
        assert f"Cannot convert video {video} duration to integer"
    duration = frame_count/fps
    print(f'Current Video Proccessing FPS: {fps} with duration: {duration}')
    success = True
    count = 0
    batches = []
    tmp_frames = []
    while success:
        success, frame = cap.read()
        if success:
            if len(batches) == batch_size:
                yield batches
                batches = []
            else:
                batches.append(np.array(frame).astype(np.float32))
            count += 1
    if batches:
        # return remaining batches
        yield batches


def save_frames(video: str, destination: str):
    frames, fps = analyze_video(video)
    for idx, frame in enumerate(frames):
        Image.fromarray(frame).save(
            os.path.join(destination, f'frame_{idx}_{fps}.png'))


def parse_arguments() -> argparse.Namespace:
    """
    Returns:
        (argparse.Namespace): Returns the parsed args of the parser
    """
    epilog = """
        python3 deep_feature_extraction extract
        python3 deep_feature_extraction extractVisual
        python3 deep_feature_extraction extractAural

        """
    parser = argparse.ArgumentParser(description="Video Summarization application",
                                     formatter_class=argparse.RawTextHelpFormatter, epilog=epilog)

    tasks = parser.add_subparsers(
        title="subcommands", description="available tasks", dest="task", metavar="")

    extract = tasks.add_parser(
        "extract", help=" Extract the deep video features")
    extract.add_argument("-l", "--layers", nargs='?', default=-1, type=int,
                         help="Number of Layers to exclude from the pretrained model")
    extract.add_argument("-m", "--model", nargs='?', default='vgg19', type=str,
                         help="The pretrained model")
    extract.add_mutually_exclusive_group(required=True)
    extract.add_argument("-v", "--video", required=False,
                         help="Video Input File")
    extract.add_argument("-d", "--dir", required=False,
                         help="Videos Input Directory")

    '''
    visual_extraction = tasks.add_parser("extractVisual", help=" Extract only the visual deep video features")
    visual_extraction.add_argument("-l", "--layers", nargs='?', default=-1, type=int,
                                   help="Number of Layers to exclude from the pretrained model")
    visual_extraction.add_mutually_exclusive_group(required=True)
    visual_extraction.add_argument("-v", "--video", required=False, help="Video Input File")
    visual_extraction.add_argument("-d", "--dir", required=False, help="Videos Input Directory")

    aural_extraction = tasks.add_parser("extractAural", help=" Extract only the aural deep video features")
    aural_extraction.add_argument("-l", "--layers", nargs='?', default=-1, type=int,
                                  help="Number of Layers to exclude from the pretrained model")
    aural_extraction.add_mutually_exclusive_group(required=True)
    aural_extraction.add_argument("-v", "--video", required=False, help="Video Input File")
    aural_extraction.add_argument("-d", "--dir", required=False, help="Videos Input Directory")
    '''

    return parser.parse_args()
