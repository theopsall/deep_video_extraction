import featureExtraction as fE
from cli import parse_arguments
from utils.utils import create_dir, is_dir


def main():
    """Console script for deep_video_extraction."""
    args = parse_arguments()

    if args.task == "extractVisual" or args.task == "extractAural":
        if not is_dir(args.input):
            raise Exception("Videos directory not found!")
        if not is_dir(args.output):
            print(f'Output directory not found, trying to create it')
            create_dir(args.output)
        if not args.model:
            print(f"Model is empty, using default [vgg]")
            args.model = "vgg"

        if args.task == "extractVisual":
            fE.extract_visual_features(
                args.input,
                model=args.model,
                layers=args.layers,
                flatten=args.flatten,
                save=args.store,
                output=args.output,
            )
        elif args.task == "extractAural":
            fE.extract_aural_features(
                args.input,
                model=args.model,
                layers=args.layers,
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


if __name__ == "__main__":
    main()
