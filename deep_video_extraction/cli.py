"""Console script for deep_video_extraction."""
import argparse

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
    parser = argparse.ArgumentParser(description="Video Summarization application",
                                     formatter_class=argparse.RawTextHelpFormatter, epilog=epilog)

    tasks = parser.add_subparsers(
        title="subcommands", description="available tasks", dest="task", metavar="")

    extract = tasks.add_parser(
        "extract", help=" Extract deep video features")
    extract.add_argument("-l", "--layers", nargs='?', default=-1, type=int,
                         help="Number of Layers to exclude from the pretrained model")
    # extract.add_argument("-m", "--model", nargs='?', default='vgg19', type=str,
    #                      help="The pretrained model")
    # extract.add_mutually_exclusive_group(required=True)
    # extract.add_argument("-v", "--video", required=False,
    #                      help="Video Input File")
    # extract.add_argument("-d", "--dir", required=False,
    #                      help="Videos Input Directory")

    '''
    visual = tasks.add_parser("visual", help=" Extract only the visual deep video features")
    visual.add_argument("-l", "--layers", nargs='?', default=-1, type=int,
                                   help="Number of Layers to exclude from the pretrained model")
    visual.add_mutually_exclusive_group(required=True)
    visual.add_argument("-v", "--video", required=False, help="Video Input File")
    visual.add_argument("-d", "--dir", required=False, help="Videos Input Directory")

    aural = tasks.add_parser("aural", help=" Extract only the aural deep video features")
    aural.add_argument("-l", "--layers", nargs='?', default=-1, type=int,
                                  help="Number of Layers to exclude from the pretrained model")
    aural.add_mutually_exclusive_group(required=True)
    aural.add_argument("-v", "--video", required=False, help="Video Input File")
    aural.add_argument("-d", "--dir", required=False, help="Videos Input Directory")
    '''

    return parser.parse_args()

def main():
    """Console script for deep_video_extraction."""
    args =  parse_arguments()

    if args.task == "extract":
        
        args.video
        if is_dir(args.video):

            pass
        elif not not args.video:
            pass

    elif args.task == "extractVisual":
        if not not args.video:
            pass
        elif not not args.video:
            pass
    elif args.task == "extractAural":
        if not not args.video:
            pass
        elif not not args.video:
            pass
    else:
        print(f'Task {args.task} not Found. Please check the script description with --help option')
    return 0



