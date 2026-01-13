#!/usr/bin/env python3
"""Plot chi slice CSVs exported by bowen_york_fast as a tiled figure."""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from plot_rk4_slices import load_slice, reshape_to_grid  # type: ignore


STEP_RE = re.compile(r"step(\d+)")


def discover_slice_files(root: Path, pattern: str) -> List[Path]:
    files = sorted(root.glob(pattern))
    return files


def extract_step(path: Path) -> int | None:
    match = STEP_RE.search(path.stem)
    if not match:
        return None
    return int(match.group(1))


def filter_steps(files: Iterable[Path], steps: List[int] | None) -> List[Path]:
    if not steps:
        return list(files)
    steps_set = set(steps)
    filtered = [f for f in files if extract_step(f) in steps_set]
    missing = steps_set - {extract_step(f) for f in filtered}
    if missing:
        raise ValueError(f"Could not find slice files for steps: {sorted(missing)}")
    return filtered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        help="Directory containing chi_z_step*.csv files (e.g. Output/rk4/bowen_york_fast)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="chi_z_step*.csv",
        help="Glob pattern used to find slice CSV files",
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="*",
        help="Optional list of step numbers to plot (defaults to all discovered)",
    )
    parser.add_argument(
        "--save",
        type=Path,
        help="Optional output image path; if omitted the plot will be displayed interactively",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=0,
        help="Number of subplot columns (auto if not provided)",
    )
    args = parser.parse_args()

    if not args.root.exists():
        parser.error(f"Directory {args.root} does not exist")

    candidates = discover_slice_files(args.root, args.pattern)
    if not candidates:
        parser.error(f"No files matched pattern {args.pattern!r} in {args.root}")

    selected = filter_steps(candidates, args.steps)
    selected.sort(key=lambda p: extract_step(p) or -1)

    fields = []
    global_min = np.inf
    global_max = -np.inf
    for csv_path in selected:
        x_vals, y_vals, values, value_name = load_slice(csv_path)
        X, Y, grid = reshape_to_grid(x_vals, y_vals, values)
        fields.append((csv_path, value_name, X, Y, grid))
        global_min = min(global_min, float(grid.min()))
        global_max = max(global_max, float(grid.max()))

    n = len(fields)
    cols = args.cols if args.cols > 0 else math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)

    for ax, entry in zip(axes.flat, fields):
        csv_path, value_name, X, Y, grid = entry
        step = extract_step(csv_path)
        mesh = ax.pcolormesh(
            X,
            Y,
            grid,
            shading="auto",
            vmin=global_min,
            vmax=global_max,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{value_name} step {step}")
        fig.colorbar(mesh, ax=ax)

    # Hide unused axes if any
    for ax in axes.flat[n:]:
        ax.set_visible(False)

    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"Saved figure to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
