import argparse
import jax.numpy as np
from flux import Flux

PI = np.pi


def get_args(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--poisson_dir",
        help="",
        default="",
        type=str,
    )
    parser.add_argument(
        "--read_write_dir",
        help="",
        default="",
        type=str,
    )
    
    if argv is not None:
        return parser.parse_args(argv)
    else:
        return parser.parse_args()
