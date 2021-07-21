import argparse
import os
import numpy as np
import cv2

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mkv', 'webm'}


def parse_arguments() -> argparse.Namespace:
    """
    Command line argument parser
    Returns:
        parser: Argument Parser
    """
    visual_extraction = argparse.ArgumentParser(description="Visual Deep Feature Extraction")

    visual_extraction.add_argument("-a", "--audio", help="Input audio data")
    visual_extraction.add_argument("-g", "--groundtruth", help="Ground truth data")
    visual_extraction.add_argument("-o", "--output",
                                   default='smoted_svc',
                                   help="Output filename for classifier")

    visual_extraction.add_argument("-res", "--resampled",
                                   default=2000, type=int,
                                   help="Number of resampled data")

    return visual_extraction.parse_args()


def is_dir(directory: str) -> bool:
    return os.path.isdir(directory)


def is_file(filename: str) -> bool:
    pass


def is_video(video: str) -> bool:
    pass


def crawl_directory(directory: str) -> list:
    if not is_dir(directory):
        raise FileNotFoundError

    tree = []
    subdirs = [folder[0] for folder in os.walk(directory)]

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

def analyze_video(video: str, batch_size: int) -> np.ndarray:
    cap = cv2.VideoCapture(video)
    try:
        fps = int(cap.get(cv2.CAP_PROP_FPS))  # frame per second for the current video in order to average the frames
    except ValueError:
        assert f"Cannot convert video {video} fps to integer"

    success = True
    count = 0
    batches = []
    average_frames = []
    tmp_frames = []
    while success:
        success, frame = cap.read()
        if (count + 1) % (fps + 1) == 0:
            batches.append(np.average(tmp_frames, axis=0).transpose())
            tmp_frames = []
            if len(batches) == batch_size:
                yield np.array(batches)
                batches = []
        else:
            tmp_frames.append(frame)
        count += 1
    if batches:
        # return remaining batches
        yield np.average(tmp_frames, axis=0).transpose()
    # return np.array(average_frames)