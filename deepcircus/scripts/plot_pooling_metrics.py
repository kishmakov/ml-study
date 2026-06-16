#!/usr/bin/env python3
"""Plot DeepCircus pooling RMSE and MAE metrics."""

from __future__ import annotations

from argparse import ArgumentParser
from json import load
from os import environ, makedirs
from os.path import dirname, join
from pathlib import Path
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [path for path in sys.path if Path(path or ".").resolve() != SCRIPT_DIR]

environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
makedirs(environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib.pyplot as plt


DEFAULT_META = "/tmp/circus/experiment_pooling.json"
GROUPS = (
    (4, 10),
    (11, 18),
    (19, 25),
    (26, 32),
)
COLORS = (
    (0, 0, 0),
    (105, 105, 105),
    (139, 0, 0),
    (255, 0, 0),
    (0, 0, 139),
    (0, 0, 255),
    (0, 100, 0),
    (34, 139, 34),
)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("meta_json", nargs="?", default=DEFAULT_META)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    with open(args.meta_json, encoding="utf-8") as f:
        meta = load(f)

    output_dir = args.output_dir or dirname(args.meta_json) or "."
    makedirs(output_dir, exist_ok=True)

    for start, end in GROUPS:
        out_path = join(output_dir, f"pooling_metrics_b{start:02d}_b{end:02d}.png")
        plot_group(meta, start, end, out_path)
        print(out_path)


def plot_group(meta: dict[str, Any], start: int, end: int, out_path: str) -> None:
    metrics = meta.get("metrics", [])
    series = {
        bitness: {"rmse": [], "mae": []}
        for bitness in range(start, end + 1)
    }

    for metric in metrics:
        bitness = int(metric["bitness"])
        if bitness < start or bitness > end:
            continue
        iteration = float(metric["iteration"])
        series[bitness]["rmse"].append((iteration, float(metric["rmse"])))
        series[bitness]["mae"].append((iteration, float(metric["mae"])))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150)
    for idx, bitness in enumerate(range(start, end + 1)):
        color = tuple(channel / 255 for channel in COLORS[idx % len(COLORS)])
        rmse_points = series[bitness]["rmse"]
        mae_points = series[bitness]["mae"]
        if rmse_points:
            axes[0].plot(
                *zip(*rmse_points),
                label=str(bitness),
                color=color,
                linewidth=2,
            )
        if mae_points:
            axes[1].plot(
                *zip(*mae_points),
                label=str(bitness),
                color=color,
                linewidth=2,
            )

    axes[0].set_ylabel("RMSE")
    axes[1].set_ylabel("MAE")
    fig.suptitle(f"Pooling validation metrics, bitness {start}-{end}")
    for axis in axes:
        axis.set_xlabel("Iteration")
        axis.grid(alpha=0.2)
        if len(axis.lines):
            axis.legend(title="Bitness", ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
