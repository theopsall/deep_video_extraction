import os
import pathlib
import cv2

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mkv', 'webm'}

def is_dir(directory: str)->bool:
    return os.path.isdir(directory)

def is_file():
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

def allowed_file(filename:str):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def isolate_audio(path: str):
    pass

def get_spectrogram(audio):
    pass