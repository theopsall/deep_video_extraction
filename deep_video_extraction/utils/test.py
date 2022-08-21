from functools import wraps
from time import time

import cv2
import numpy as np

EXPORT_ALL = "all"
EXPORT_LAST = "last"
EXPORT_FIRST = "first"
EXPORT_RANDOM = "random"


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time()
        func(*args, **kwargs)
        end = time()
        print(f"{func.__name__} took {end - start} seconds")

    return wrapper


def analyze_video(video: str, keep: str = "last") -> np.ndarray:
    cap = cv2.VideoCapture(video)
    try:
        # frame per second for the current video in order to average the frames
        _FPS = int(cap.get(cv2.CAP_PROP_FPS))
        fps = _FPS + 1
        print(fps)
        print(_FPS)
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


if __name__ == "__main__":
    video = "/media/theo/Data/Git/deep_video_extraction/deep_video_extraction/example/SampleVideo.mp4"
    x = analyze_video(video)
    print(len(x))
    print(x.shape)
