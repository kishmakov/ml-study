"""Plot cost-to-go training metrics from ``meta_*.json``.

Expected metric shape:

    {
      "config": {
        "ctg_eval": {
          "max_states": 6500000,
          "metrics": [
            {
              "iteration": 1,
              "depths": {
                "1": {"percent_solved": 100.0, "avg_cost_to_go": 0.9},
                "5": {"percent_solved": 90.0, "avg_cost_to_go": 4.2}
              }
            }
          ]
        }
      }
    }
"""

from __future__ import annotations

from argparse import ArgumentParser
from json import load
from os import environ, makedirs
from os.path import basename, dirname, join, splitext
from typing import Any

environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
makedirs(environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib.pyplot as plt


COLORS = (
    (0, 0, 0),
    (105, 105, 105),
    (205, 205, 205),
    (139, 0, 0),
    (255, 0, 0),
    (250, 128, 114),
    (0, 0, 139),
    (0, 0, 255),
    (173, 216, 230),
)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("meta_json", nargs="+")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    assert args.output is None or len(args.meta_json) == 1, (
        "--output can only be used with one meta file"
    )

    for meta_path in args.meta_json:
        with open(meta_path, encoding="utf-8") as f:
            meta = load(f)
        series = load_ctg_series(meta)
        out_path = args.output or ctg_path_for(meta_path)
        plot_ctg(series, out_path)
        print(out_path)


def ctg_path_for(meta_path: str) -> str:
    stem = splitext(basename(meta_path))[0]
    if stem.startswith("meta_"):
        stem = "ctg_" + stem.removeprefix("meta_")
    else:
        stem = stem + "_ctg"
    return join(dirname(meta_path), stem + ".png")


def load_ctg_series(meta: dict[str, Any]) -> dict[str, Any]:
    config = meta["config"]
    assert "ctg_eval" in config, "No config.ctg_eval found in meta json"
    ctg_eval = config["ctg_eval"]
    metrics = ctg_eval["metrics"]
    assert metrics, "No config.ctg_eval.metrics found in meta json"

    depths = sorted(int(depth) for depth in metrics[0]["depths"])
    percent: dict[int, list[tuple[float, float]]] = {depth: [] for depth in depths}
    avg_ctg: dict[int, list[tuple[float, float]]] = {depth: [] for depth in depths}

    for idx, metric in enumerate(metrics):
        iteration = float(metric.get("iteration", idx + 1))
        by_depth = metric["depths"]
        for depth in depths:
            values = by_depth[str(depth)]
            percent[depth].append((iteration, float(values["percent_solved"])))
            avg_ctg[depth].append((iteration, float(values["avg_cost_to_go"])))

    return {
        "depths": depths,
        "percent_solved": percent,
        "avg_cost_to_go": avg_ctg,
        "max_states": ctg_eval.get("max_states"),
    }


def plot_ctg(series: dict[str, Any], out_path: str) -> None:
    plot_ctg_matplotlib(series, out_path)


def plot_ctg_matplotlib(series: dict[str, Any], out_path: str) -> None:
    depths = series["depths"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150)
    for idx, depth in enumerate(depths):
        color = tuple(channel / 255 for channel in COLORS[idx % len(COLORS)])
        points = series["percent_solved"][depth]
        if points:
            axes[0].plot(*zip(*points), label=str(depth), color=color, linewidth=2)
        points = series["avg_cost_to_go"][depth]
        if points:
            axes[1].plot(*zip(*points), label=str(depth), color=color, linewidth=2)
            axes[1].axhline(depth, color=color, linestyle="--", linewidth=1)

    axes[0].set_ylabel("Percent solved")
    axes[1].set_ylabel("Average cost-to-go")
    max_states = series.get("max_states")
    if max_states is not None:
        fig.suptitle(f"CTG evaluation max states: {int(max_states):,}")
    for axis in axes:
        axis.set_xlabel("Iteration")
        axis.legend(title="No. of scrambles", ncol=3)
        axis.grid(alpha=0.2)
    fig.tight_layout(rect=(0, 0, 1, 0.94) if max_states is not None else None)
    fig.savefig(out_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
