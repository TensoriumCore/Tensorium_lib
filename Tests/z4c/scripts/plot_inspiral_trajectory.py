#!/usr/bin/env python3
"""Quick visualizer for Output/rk4/bowen_york/trajectory.csv."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_trajectory(path: Path):
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"No samples found in {path}")
    to_float = lambda key: [float(row[key]) for row in rows]
    t = to_float("t")
    x1 = to_float("x1")
    y1 = to_float("y1")
    x2 = to_float("x2")
    y2 = to_float("y2")
    dist = to_float("distance")
    return t, x1, y1, x2, y2, dist


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path, help="trajectory CSV (t,x1,y1,x2,y2,distance)")
    args = parser.parse_args()

    t, x1, y1, x2, y2, dist = load_trajectory(args.trajectory)

    fig, (ax_orbit, ax_dist) = plt.subplots(1, 2, figsize=(10, 4.5))

    ax_orbit.plot(x1, y1, label="BH1")
    ax_orbit.plot(x2, y2, label="BH2")
    ax_orbit.set_xlabel("x")
    ax_orbit.set_ylabel("y")
    ax_orbit.set_title("Puncture Trajectories")
    ax_orbit.axis("equal")
    ax_orbit.legend()

    ax_dist.plot(t, dist, color="tab:red")
    ax_dist.set_xlabel("t")
    ax_dist.set_ylabel("Separation")
    ax_dist.set_title("BH separation vs time")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
