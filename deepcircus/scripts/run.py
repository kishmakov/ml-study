#!/usr/bin/env python3

from argparse import ArgumentParser
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent

sys.path = [path for path in sys.path if Path(path or ".").resolve() != SCRIPT_DIR]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("experiment", choices=("depth", "pooling"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from generator.generator import load_generator

    generator = load_generator()

    if args.experiment == "depth":
        from experiments.experiment_depth import run_experiment
    else:
        from experiments.experiment_pooling import run_experiment

    run_experiment(generator)


if __name__ == "__main__":
    main()
