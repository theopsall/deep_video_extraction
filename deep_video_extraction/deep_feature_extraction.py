from deep_video_extraction.utils.utils import parse_arguments, is_dir


def main():
    args = parse_arguments()
    if args.task == "extract":
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


if __name__ == "__main__":
    main()
