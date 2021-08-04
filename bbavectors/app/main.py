import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='BBAVectors')
    parser.add_argument('weights_dir', type=str, help="weights directory")
    parser.add_argument('image_path', type=str, help="image path")
    parser.add_argument('drone_altitude', type=int, help="drone altitude")
    parser.add_argument('--plot', dest='plot', action='store_true')
    parser.add_argument('--cpu', dest='cpu', action='store_true')
    parser.add_argument('--no-docker', dest='nodocker', action='store_true')
    return parser.parse_args()


def main():
    """BBAVectors Inference CLI.

    Usage:
        bbavectors [-h] weights_dir image_path drone_altitude [OPTIONS]

    Options:
        -h, --help      Show this message.
    """

    args = parse_args()

    use_gpu = "" if args.cpu else "--gpus=all"
    image_dir = os.path.dirname(args.image_path)
    image_name = os.path.basename(args.image_path)
    altitude = int(args.drone_altitude)
    results_path = os.getcwd()

    if args.nodocker:
        from bbavectors.app import run_inference
        run_inference(
            args.weights_dir, args.image_path,
            altitude, args.plot
        )
    else:
        command = f"""
            docker run --rm -it {use_gpu}
            -v "{image_dir}":/image_dir/
            -v "{args.weights_dir}":/weights_dir/
            -v "{results_path}":/results/ \
            bbavectors
            bbavectors
                /weights_dir/"{args.weights_dir}"
                /image_dir/"{image_name}"
                {altitude} --plot --no-docker
        """
        os.system(command)


if __name__ == "__main__":
    main()
