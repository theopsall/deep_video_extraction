import cv2
import numpy as np


def read(video: str) -> np.ndarray:
    cap = cv2.VideoCapture(video)
    try:
        fps = int(cap.get(cv2.CAP_PROP_FPS))  # frame per second for the current video in order to average the frames
    except ValueError:
        assert f"Cannot convert video {video} fps to integer"

    success = True
    count = 0
    average_frames = []
    tmp_frames = []
    while success:
        success, frame = cap.read()
        if (count + 1) % (fps + 1) == 0:

            average_frames.append(np.average(tmp_frames, axis=0))
            tmp_frames = []
        else:
            tmp_frames.append(frame)
        count += 1

    return np.array(average_frames)


def main():
    pass


if __name__ == "__main__":
    average = read('videos/samples.mp4')
    print(type(average))
