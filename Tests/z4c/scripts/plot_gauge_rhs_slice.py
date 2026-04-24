#!/usr/bin/env python3
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_slice(path: Path):
    df = pd.read_csv(path)
    xs = np.sort(df.iloc[:, 0].unique())
    ys = np.sort(df.iloc[:, 1].unique())
    values = df.iloc[:, 2].values.reshape((len(xs), len(ys)))
    colname = df.columns[2]
    return xs, ys, values, colname


def main():
    default = Path("rhs_alpha_slice_schwarzschild.csv")
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    if not path.exists():
        print(
            f"Slice '{path}' not found. Run TensoriumGaugeEvolutionDemo to generate the CSV.",
            file=sys.stderr)
        sys.exit(1)

    xs, ys, grid, colname = load_slice(path)

    fig, ax = plt.subplots(figsize=(6, 5))
    mesh = ax.pcolormesh(ys, xs, grid, shading="auto", cmap="coolwarm")
    ax.set_xlabel("y")
    ax.set_ylabel("x")
    ax.set_title(f"Gauge RHS slice: {path.name}")
    fig.colorbar(mesh, ax=ax, label=colname)
    fig.tight_layout()

    out_path = path.with_name(path.stem + "_plot.png")
    fig.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
