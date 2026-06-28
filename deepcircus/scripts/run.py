#!/usr/bin/env python3

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEEPCIRCUS_DIR = Path(__file__).resolve().parents[1]

sys.path = [path for path in sys.path if Path(path or ".").resolve() != SCRIPT_DIR]
sys.path.insert(0, str(DEEPCIRCUS_DIR))
sys.path.insert(0, str(DEEPCIRCUS_DIR / "bool-bench"))


def main() -> None:
    from experiments.bitness import run_training
    from experiments.generator_proxy import GeneratorProxy

    generator = GeneratorProxy(16)
    try:
        run_training(generator)
    finally:
        generator.close()


if __name__ == "__main__":
    main()
