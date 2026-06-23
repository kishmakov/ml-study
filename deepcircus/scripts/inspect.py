#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = str(Path(__file__).resolve().parent)
sys.path = [path for path in sys.path if path != SCRIPT_DIR]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generator.generator import load_generator


def input_bits(input_id: int, bitness: int) -> str:
    return "".join("1" if (input_id >> bit) & 1 else "0" for bit in range(bitness))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bitness", type=int, nargs="?", default=2)
    parser.add_argument("case_id", type=int, nargs="?", default=6)
    args = parser.parse_args()

    assert 0 <= args.bitness <= 4, "small bitness must be in [0, 4]"

    generator = load_generator()
    cases_number = generator.cases_number(args.bitness)
    assert 0 <= args.case_id < cases_number, "case_id is out of range"

    print(
        f"bitness={args.bitness} "
        f"case={args.case_id} "
        f"nodes={generator.case_nodes(args.bitness, args.case_id)} "
        f"active={generator.case_active_bits(args.bitness, args.case_id)}"
    )

    for input_id in range(1 << args.bitness):
        bits = input_bits(input_id, args.bitness)
        value = generator.case_value(args.bitness, args.case_id, bits)
        print(f"{bits} -> {int(value[args.bitness] > 0)}")


if __name__ == "__main__":
    main()
