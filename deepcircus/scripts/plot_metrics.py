#!/usr/bin/env python3
"""Plot DeepCircus experiment metrics from self-contained JSON metadata."""

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


DEFAULT_META = "/tmp/circus/experiment.json"
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

    for series in meta["series"]:
        out_path = join(output_dir, f"{series['name']}.png")
        plot_series(meta["metrics"], series, out_path)
        print(out_path)


def plot_series(
    metrics: list[dict[str, Any]],
    series: dict[str, Any],
    out_path: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150)

    for line_id, line in enumerate(series["lines"]):
        points = line_points(metrics, line)
        if not points:
            continue
        color = plot_color(line, line_id)
        x_values = [point["x"] for point in points]
        mae_values = [point["mae"] for point in points]
        rmse_values = [point["rmse"] for point in points]

        axes[0].plot(
            x_values,
            mae_values,
            label=line["label"],
            color=color,
            linewidth=float(line.get("linewidth", 2)),
        )
        axes[1].plot(
            x_values,
            rmse_values,
            label=line["label"],
            color=color,
            linewidth=float(line.get("linewidth", 2)),
        )

    axes[0].set_ylabel("MAE")
    axes[1].set_ylabel("RMSE")
    fig.suptitle(series["title"])
    for axis in axes:
        axis.set_xlabel(series["x_label"])
        axis.grid(alpha=0.2)
        if len(axis.lines):
            axis.legend(title=series.get("legend_title"))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path)
    plt.close(fig)


def line_points(metrics: list[dict[str, Any]], line: dict[str, Any]) -> list[dict[str, float]]:
    x_key = line["x_key"]
    mae_key = line["mae_key"]
    rmse_key = line["rmse_key"]
    where = line.get("where", {})

    points = []
    for metric in metrics:
        if not matches_where(metric, where):
            continue
        if x_key not in metric or mae_key not in metric or rmse_key not in metric:
            continue
        points.append(
            {
                "x": float(metric[x_key]),
                "mae": float(metric[mae_key]),
                "rmse": float(metric[rmse_key]),
            }
        )

    points.sort(key=lambda point: point["x"])
    return points


def plot_color(line: dict[str, Any], line_id: int) -> tuple[float, float, float]:
    color = line.get("color", COLORS[line_id % len(COLORS)])
    return tuple(channel / 255 for channel in color)


def matches_where(metric: dict[str, Any], where: dict[str, Any]) -> bool:
    for key, expected in where.items():
        if metric.get(key) != expected:
            return False
    return True


if __name__ == "__main__":
    main()
