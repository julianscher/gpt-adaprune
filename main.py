from src.utils.arg_parser import ArgParser, load_config
from src.utils.worker import Worker


def main() -> None:
    parser = ArgParser()
    args = parser.parse()
    args = load_config(args, parser)

    worker = Worker(args, parser)
    worker.run()


if __name__ == "__main__":
    main()
