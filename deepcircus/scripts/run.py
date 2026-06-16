#!/usr/bin/env python3

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent

sys.path = [path for path in sys.path if Path(path or ".").resolve() != SCRIPT_DIR]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.experiment_pooling import run_experiment
from experiments.generator import load_generator


def main() -> None:
    generator = load_generator()
    run_experiment(generator)


if __name__ == "__main__":
    main()
