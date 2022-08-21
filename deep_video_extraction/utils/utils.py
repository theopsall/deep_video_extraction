import argparse
import os
import random
from functools import wraps
from subprocess import PIPE, Popen
from time import time
from xmlrpc.client import Boolean

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
from scipy import signal
from scipy.io import wavfile

from utils._types import *

ALLOWED_EXTENSIONS = {"mp4", "avi", "mkv", "webm"}


def device():
    """
    Check if cuda is avaliable else choose the cpu
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu') # For testing purposes
    print(f"pyTorch is using {device}")
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


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time()
        func(*args, **kwargs)
        end = time()
        print(f"{func.__name__} took {end - start} seconds")

    return wrapper


def is_dir(directory: str) -> bool:
    return os.path.isdir(directory)


def is_dir_empty(directory: str) -> bool:
    return len(os.listdir(directory)) == 0


def create_dir(directory: str) -> bool:
    try:
        return os.makedirs(directory)
    except FileExistsError:
        print(f"{directory} already exists")
        return False


def is_file(filename: str) -> bool:
    return os.path.isfile(filename)


def is_video(video: str) -> bool:
    return video.split(".")[-1] in ALLOWED_EXTENSIONS


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


def sound_isolation(videopath: str, audio_output: str) -> bool:
    """
    Isolate the audio signal from a video stream
    in wav file with sampling rate= 1600 in mono channel

    Args:
        video (str): videopath

    Returns:
        bool: True if audio isolated successfully, False otherwise
    """
    command = "ffmpeg -i '{0}' -q:a 0 -ac 1 -ar 16000  -map a '{1}'".format(
        videopath, audio_output
    )
    try:
        os.system(command)
        return True
    except:
        print("Audio isolation failed")
        return False


def clone_structure(src: str, dst: str) -> None:
    pass


def read_video():
    pass


def allowed_file(filename: str):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_timestamp():
    return f"{time()}"


def isolate_audio(path: str):
    pass


def get_spectrogram(audio):
    pass


def analyze_video(video: str, keep: str = "last") -> np.ndarray:
    cap = cv2.VideoCapture(video)
    try:
        # frame per second for the current video in order to average the frames
        _FPS = int(cap.get(cv2.CAP_PROP_FPS))
        fps = _FPS + 1
    except ValueError:
        assert f"Cannot convert video {video} fps to integer"
    # print(f'Proccessing {video} with: {fps} fps')
    success = True
    batches = []

    count = 0

    while success:
        success, frame = cap.read()
        if success:
            frame = cv2.resize(frame, (124, 124))
            if keep == EXPORT_FIRST:
                if count % fps == 0:
                    batches.append(np.array(frame))
            if keep == EXPORT_LAST:
                if count % fps == _FPS:
                    batches.append(np.array(frame))
            if keep == EXPORT_ALL:
                if count % fps == _FPS:
                    batches.append(np.array(frame))
        count += 1
    return np.array(batches)


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
    duration = frame_count / fps
    print(f"Current Video Proccessing FPS: {fps} with duration: {duration}")
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
        Image.fromarray(frame).save(os.path.join(destination, f"frame_{idx}_{fps}.png"))


def stereo_to_mono(signal):
    """
    Input signal (stored in a numpy array) to MONO (if it is STEREO)
    """

    if signal.ndim == 2:
        if signal.shape[1] == 1:
            signal = signal.flatten()
        else:
            if signal.shape[1] == 2:
                signal = (signal[:, 1] / 2) + (signal[:, 0] / 2)
    return signal


def get_spectrogram(audio: str, output: str) -> None:
    """
    Creates a spectrogram of the audio file
    Args:
        audio (str): The path to the audio file
        output (str): The path to the output directory
    """
    fs, data = wavfile.read(audio)
    data = stereo_to_mono(data)
    fig, ax = plt.subplots(1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis("off")
    pxx, freqs, bins, im = ax.specgram(x=data, Fs=fs, noverlap=384, NFFT=512)
    ax.axis("off")
    fig.savefig(output, dpi=100, frameon="false")
    fig.clf()
