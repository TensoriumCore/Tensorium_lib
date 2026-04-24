#!/usr/bin/env python3
"""Visualize BSSN RK4 slice CSV files with an interactive pcolormesh plot."""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


def load_slice(csv_path: Path):
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    if data.size == 0:
        raise ValueError(f"No data found in {csv_path}")
    if data.ndim == 0:
        # Promote scalar rows to 1-element arrays
        data = np.array([data], dtype=data.dtype)

    x_vals = data["x"].astype(float)
    y_vals = data["y"].astype(float)
    value_fields = [
        name for name in data.dtype.names if name not in {"x", "y"}]
    if not value_fields:
        raise ValueError(f"Could not locate value column in {csv_path}")
    value_name = value_fields[0]
    values = data[value_name].astype(float)
    return x_vals, y_vals, values, value_name


def reshape_to_grid(x_vals, y_vals, values):
    unique_x = np.unique(x_vals)
    unique_y = np.unique(y_vals)
    nx = unique_x.size
    ny = unique_y.size
    if nx * ny != values.size:
        # Fallback: enforce lexicographic ordering prior to reshaping
        idx = np.lexsort((x_vals, y_vals))
        x_vals = x_vals[idx]
        y_vals = y_vals[idx]
        values = values[idx]
        unique_x = np.unique(x_vals)
        unique_y = np.unique(y_vals)
        nx = unique_x.size
        ny = unique_y.size
    grid = values.reshape((ny, nx))
    X, Y = np.meshgrid(unique_x, unique_y)
    return X, Y, grid


def plot_slice(csv_path: Path, title: Optional[str] = None):
    x_vals, y_vals, values, value_name = load_slice(csv_path)
    X, Y, grid = reshape_to_grid(x_vals, y_vals, values)
    rms = float(np.sqrt(np.mean(values ** 2)))

    print(
        f"{csv_path}: min={values.min():.3e} max={values.max():.3e} rms={rms:.3e}",
        file=sys.stdout,
    )

    plt.figure(figsize=(6, 5))
    mesh = plt.pcolormesh(X, Y, grid, shading="auto")
    plt.colorbar(mesh, label=value_name)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title or f"{value_name} slice")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="Path to a slice CSV file")
    parser.add_argument(
        "--title", type=str, default=None, help="Optional plot title override"
    )
    args = parser.parse_args()

    plot_slice(args.csv_path, args.title)


if __name__ == "__main__":
    main()
