import argparse
import os

import cv2
import numpy as np

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mkv', 'webm'}


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

    tasks = parser.add_subparsers(title="subcommands", description="available tasks", dest="task", metavar="")

    extract = tasks.add_parser("extract", help=" Extract the deep video features")
    extract.add_argument("-l", "--layers", nargs='?', default=-1, type=int,
                         help="Number of Layers to exclude from the pretrained model")
    extract.add_argument("-m", "--model", nargs='?', default='vgg19', type=str,
                         help="The pretrained model")
    extract.add_mutually_exclusive_group(required=True)
    extract.add_argument("-v", "--video", required=False, help="Video Input File")
    extract.add_argument("-d", "--dir", required=False, help="Videos Input Directory")

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

    for subdir in subdirs:
        files = next(os.walk(subdir))[2]
        for _file in files:
            yield os.path.join(subdir, _file)


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
