#!/usr/bin/env python3
import argparse
import os
import sys

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib-cache"

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot BSSN constraint norms exported during moving puncture runs."
    )
    parser.add_argument(
        "max_step",
        nargs="?",
        type=int,
        default=None,
        help="Optional maximum step to display (e.g. `python3 plot_constraints.py 500`).",
    )
    parser.add_argument(
        "--csv",
        default="Output/viz/constraints_norms.csv",
        help="Path to constraints CSV file.",
    )
    parser.add_argument(
        "--out",
        default="Output/viz/constraints_norms.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--x",
        choices=["step", "t"],
        default="t",
        help="X axis for the plots.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively after saving.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.csv):
        print(f"[ERR] Missing file: {args.csv}")
        return 1

    df = pd.read_csv(args.csv)
    if df.empty:
        print(f"[ERR] Empty CSV: {args.csv}")
        return 1

    required = [
        "step",
        "t",
        "l2_theta",
        "l2_Z",
        "l2_H",
        "l2_M",
        "max_H",
        "max_det_drift",
        "max_trace_A",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[ERR] Missing columns in CSV: {', '.join(missing)}")
        return 1

    if args.max_step is not None:
        df = df[df["step"] <= args.max_step].copy()
        if df.empty:
            print(f"[ERR] No samples with step <= {args.max_step}")
            return 1

    x = df[args.x].to_numpy()

    fig, axes = plt.subplots(3, 1, figsize=(11, 11), constrained_layout=True)

    ax = axes[0]
    ax.semilogy(x, df["l2_theta"], label=r"$||\Theta||_2$", linewidth=1.7)
    ax.semilogy(x, df["l2_Z"], label=r"$||Z||_2$", linewidth=1.7)
    ax.semilogy(x, df["l2_H"], label=r"$||H||_2$", linewidth=1.7)
    ax.semilogy(x, df["l2_M"], label=r"$||M||_2$", linewidth=1.7)
    ax.set_title("Constraint Norms (L2)")
    ax.set_ylabel("Norm")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend()

    ax = axes[1]
    ax.semilogy(x, df["max_H"], label=r"$||H||_{\infty}$", color="tab:red", linewidth=1.7)
    ax.set_title("Hamiltonian Constraint (Linf)")
    ax.set_ylabel("Norm")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend()

    ax = axes[2]
    ax.semilogy(
        x,
        df["max_det_drift"].abs(),
        label=r"$\max |\det(\tilde{\gamma})-1|$",
        color="tab:purple",
        linewidth=1.7,
    )
    ax.semilogy(
        x,
        df["max_trace_A"].abs(),
        label=r"$\max |\mathrm{tr}\tilde{A}|$",
        color="tab:green",
        linewidth=1.7,
    )
    ax.set_title("Algebraic Constraint Drifts")
    ax.set_xlabel(args.x)
    ax.set_ylabel("Magnitude")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    print(f"[OK ] saved {args.out}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)

    return 0


if __name__ == "__main__":
    sys.exit(main())
