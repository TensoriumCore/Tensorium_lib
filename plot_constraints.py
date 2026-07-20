#!/usr/bin/env python3
import argparse
import os
import sys

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib-cache"
if "--show" not in sys.argv[1:]:
    os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import pandas as pd

try:
    import h5py
except Exception:
    h5py = None


DEFAULT_CSV = "Output/viz/constraints_norms.csv"
DEFAULT_H5 = "Output/viz/constraints_norms.h5"
DEFAULT_OUT = "Output/viz/constraints_norms.png"


def option_was_provided(argv, name):
    prefix = name + "="
    return any(token == name or token.startswith(prefix) for token in argv)


def resolve_csv_dir(args):
    if not args.csv_dir:
        return args

    csv_provided = (
        option_was_provided(sys.argv[1:], "--csv")
        or option_was_provided(sys.argv[1:], "--input")
    )
    out_provided = option_was_provided(sys.argv[1:], "--out")

    if not csv_provided:
        args.input_path = os.path.join(args.csv_dir, os.path.basename(DEFAULT_CSV))
    elif not os.path.isabs(args.input_path) and not os.path.dirname(args.input_path):
        args.input_path = os.path.join(args.csv_dir, args.input_path)

    if not out_provided:
        args.out = os.path.join(args.csv_dir, os.path.basename(DEFAULT_OUT))

    return args


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
        "--input",
        "--csv",
        dest="input_path",
        default=None,
        help=(
            "Path to constraints CSV or HDF5 file. "
            "Defaults to Output/viz/constraints_norms.csv if present, else .h5."
        ),
    )
    parser.add_argument(
        "--csv-dir",
        "--data-dir",
        dest="csv_dir",
        default=None,
        help="Directory containing constraints_norms.csv.",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
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
    return resolve_csv_dir(parser.parse_args())


def default_input_path():
    if os.path.exists(DEFAULT_CSV):
        return DEFAULT_CSV
    if os.path.exists(DEFAULT_H5):
        return DEFAULT_H5
    return DEFAULT_CSV


def is_hdf5_path(path):
    return path.lower().endswith((".h5", ".hdf5"))


def load_hdf5_constraints(path):
    if h5py is None:
        raise RuntimeError(
            f"{path} is HDF5, but Python package 'h5py' is not available."
        )

    columns = {}
    with h5py.File(path, "r") as h5:
        for name, obj in h5.items():
            if not isinstance(obj, h5py.Dataset):
                continue
            values = obj[()]
            if getattr(values, "ndim", 0) == 0:
                values = [values.item()]
            else:
                values = values.reshape(-1)
            columns[name] = values

    if not columns:
        raise RuntimeError(f"No datasets found in HDF5 file: {path}")

    lengths = {name: len(values) for name, values in columns.items()}
    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        detail = ", ".join(f"{name}={length}" for name, length in sorted(lengths.items()))
        raise RuntimeError(f"HDF5 datasets have inconsistent lengths: {detail}")

    return pd.DataFrame(columns)


def load_constraints(path):
    if is_hdf5_path(path):
        return load_hdf5_constraints(path)
    return pd.read_csv(path)


def main():
    args = parse_args()
    input_path = args.input_path or default_input_path()

    if not os.path.exists(input_path):
        print(f"[ERR] Missing file: {input_path}")
        return 1

    try:
        df = load_constraints(input_path)
    except Exception as exc:
        print(f"[ERR] Could not read {input_path}: {exc}")
        return 1

    if df.empty:
        print(f"[ERR] Empty constraints file: {input_path}")
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
        print(f"[ERR] Missing columns in {input_path}: {', '.join(missing)}")
        return 1

    df = df.sort_values("step").reset_index(drop=True)
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
    print(f"[OK ] read {input_path}")
    print(f"[OK ] saved {args.out}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)

    return 0


if __name__ == "__main__":
    sys.exit(main())
