"""Console script for deep_video_extraction."""
import argparse

import featureExtraction as fE
from utils.utils import is_dir


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
    parser = argparse.ArgumentParser(
        description="Video Summarization application",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=epilog,
    )

    tasks = parser.add_subparsers(
        title="subcommands", description="available tasks", dest="task", metavar=""
    )

    extractVisual = tasks.add_parser(
        "extractVisual", help="Extract deep video features from a directory of videos"
    )
    extractVisual.add_argument(
        "-i", "--input", required=False, help="Input Directory with videos"
    )
    extractVisual.add_argument(
        "-m", "--model", nargs="?", default="vgg", type=str, help="The pretrained model"
    )
    extractVisual.add_argument(
        "-l",
        "--layers",
        nargs="?",
        default=-1,
        type=int,
        help="Number of Layers to exclude from the pretrained model",
    )
    extractVisual.add_argument(
        "-f",
        "--flatten",
        required=False,
        type=bool,
        help="Flatten last layer of feature vector",
    )
    extractVisual.add_argument(
        "-s", "--store", required=False, type=bool, help="Store feature vectors"
    )
    extractVisual.add_argument("-o", "--output", required=False, help="Output directory")
    
    extractAural = tasks.add_parser(
        "extractAural", help="Extract deep audio features from a directory of audios"
    )
    extractAural.add_argument(
        "-i", "--input", required=False, help="Input Directory with audios"
    )
    extractAural.add_argument(
        "-m", "--model", nargs="?", default="vgg", type=str, help="The pretrained model"
    )
    extractAural.add_argument(
        "-l",
        "--layers",
        nargs="?",
        default=-1,
        type=int,
        help="Number of Layers to exclude from the pretrained model",
    )
    extractAural.add_argument(
        "-f",
        "--flatten",
        required=False,
        type=bool,
        help="Flatten last layer of feature vector",
    )
    extractAural.add_argument(
        "-s", "--store", required=False, type=bool, help="Store feature vectors"
    )
    extractAural.add_argument("-o", "--output", required=False, help="Output directory")


    sound = tasks.add_parser(
        "soundIsolation", help="Isolate the audio from videos input"
    )
    sound.add_argument(
        "-i", "--input", required=False, help="Input Directory with videos"
    )
    sound.add_argument("-o", "--output", required=False, help="Output directory")

    spectro = tasks.add_parser("spectro", help="Extract spectrograms from audio files")
    spectro.add_argument(
        "-i", "--input", required=False, help="Input Directory with audio files"
    )
    spectro.add_argument("-o", "--output", required=False, help="Output directory")

    return parser.parse_args()


def main():
    """Console script for deep_video_extraction."""
    args = parse_arguments()

    if args.task == "extractVisual":
        if not is_dir(args.input):
            raise Exception("Videos directory not found!")
        elif not args.model:
            print(f"Model is empty, using default")
            args.model = "vgg"
        fE.extract_visual_features(
            args.input,
            model=args.model,
            layers=args.layers,
            flatten=args.flatten,
            save=args.store,
            output=args.output,
        )
    if args.task == "extractAural":
        if not is_dir(args.input):
            raise Exception("Audio directory not found!")
        elif not args.model:
            print(f"Model is empty, using default")
            args.model = "vgg"
        fE.extract_aural_features(
            args.input,
            model=args.model,
            layers=args.layers,
            flatten=args.flatten,
            save=args.store,
            output=args.output,
        )
    elif args.task == "soundIsolation":
        if not is_dir(args.input):
            raise Exception("Videos directory not found!")
        fE.audio_extraction(args.input)
    elif args.task == "spectro":
        if not is_dir(args.input):
            raise Exception("Videos directory not found!")
        fE.extract_spectros(args.input)
    else:
        print(
            f"Task {args.task} not Found. Please check the script description with --help option"
        )
    return 0
