import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='BBAVectors')
    parser.add_argument(
        'env', type=str, help="Execution environment [docker/local]")
    parser.add_argument('weights_dir', type=str, help="Weights directory")
    parser.add_argument('image_path', type=str, help="Image path")
    parser.add_argument('drone_altitude', type=int, help="Drone altitude")
    parser.add_argument('--plot', dest='plot', action='store_true')
    parser.add_argument('--cpu', dest='cpu', action='store_true')
    args = parser.parse_args()
    if args.env != "docker" and args.env != "local":
        parser.print_help()
        exit(1)
    return args


def main():
    """BBAVectors Inference CLI.

    usage: bbavectors [-h] [--plot] [--cpu]
                  env weights_dir image_path drone_altitude

    BBAVectors

    positional arguments:
    env             Execution environment [docker/local]
    weights_dir     Weights directory
    image_path      Image path
    drone_altitude  Drone altitude

    optional arguments:
    -h, --help      show this help message and exit
    --plot
    --cpu
    """

    args = parse_args()

    use_gpu = "" if args.cpu else "--gpus=all"
    image_dir = os.path.dirname(args.image_path)
    image_name = os.path.basename(args.image_path)
    altitude = int(args.drone_altitude)
    results_path = os.getcwd()

    if args.env == "local":
        from bbavectors.app.object_detection import run_inference
        run_inference(
            args.weights_dir, args.image_path,
            altitude, args.plot
        )
    else:
        command = (
            f'docker run --rm -it {use_gpu} '
            f'-v "{image_dir}":/image_dir/ '
            f'-v "{args.weights_dir}":/weights_dir/ '
            f'-v "{results_path}":/results/ bbavectors '
            f'bbavectors local /weights_dir/ /image_dir/"{image_name}" '
            f'{altitude} --plot'
        )
        os.system(command)


if __name__ == "__main__":
    main()
