import argparse


def add_common_args(parser):
    parser.add_argument(
        "-i", "--input", required=True, help="Input Directory"
    )
    parser.add_argument(
        "-m", "--model", nargs="?", default="vgg", type=str, help="The pretrained model"
    )
    parser.add_argument(
        "-l",
        "--layers",
        nargs="?",
        default=-1,
        type=int,
        help="Number of Layers to exclude from the pretrained model",
    )
    parser.add_argument(
        "-f",
        "--flatten",
        required=False,
        type=bool,
        help="Flatten last layer of feature vector",
    )
    parser.add_argument(
        "-s", "--store", action="store_true", help="Store feature vectors"
    )
    parser.add_argument(
        "-o", "--output", required=False, help="Output directory"
    )
    
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
    add_common_args(extractVisual)

    extractAural = tasks.add_parser(
        "extractAural", help="Extract deep audio features from a directory of audios"
    )
  
    add_common_args(extractAural)

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
