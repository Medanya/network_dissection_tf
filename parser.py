import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Run dissection method on tensorflow graph.')
    parser.add_argument(
        '--graph',
        type=str,
        default="./mobilenet/mobilenet_224.pb",
        help='path to frozen tensorflow model'
    )

    parser.add_argument(
        '--input_node',
        type=str,
        default="input",
        help='name of input node on graph'
    )

    parser.add_argument(
        '--output_node',
        type=str,
        default="MobilenetV2/Conv_1/Conv2D",
        help='name of node to analyze'
    )
    parser.add_argument(
        "--n_maps",
        default=10,
        type=int,
        help="number filters to analyze"
    )
    parser.add_argument(
        "--output_dir",
        default="./output",
        type=str,
        help="output directory"
    )
    parser.add_argument(
        "--dataset_dir",
        default="./data/",
        type=str,
        help="output directory"
    )
    parser.add_argument(
        "--percentile",
        default=99.5,
        type=float,
        help="percentile for thresholding activations"
    )


    return parser.parse_args()