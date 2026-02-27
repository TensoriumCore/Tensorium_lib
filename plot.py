import glob
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed


def _ensure_tex_binaries_in_path():
    # Keep usetex robust on macOS BasicTeX installs where texbin is not in PATH
    # for non-interactive Python launches.
    path = os.environ.get("PATH", "")
    candidates = [
        "/Library/TeX/texbin",
        "/usr/local/texlive/2025basic/bin/universal-darwin",
    ]
    for candidate in candidates:
        if not os.path.isdir(candidate):
            continue
        has_latex = os.path.isfile(os.path.join(candidate, "latex"))
        has_backend = os.path.isfile(os.path.join(candidate, "dvipng")) or os.path.isfile(
            os.path.join(candidate, "dvisvgm")
        )
        if not (has_latex and has_backend):
            continue
        parts = path.split(":") if path else []
        if candidate not in parts:
            os.environ["PATH"] = candidate + (":" + path if path else "")
        break


_ensure_tex_binaries_in_path()

raw_flags = {a for a in sys.argv[1:] if a.startswith("--")}
raw_save_png = ("--save-png" in raw_flags) or (os.getenv("TENSORIUM_PLOT_SAVE_PNG", "0") != "0")
raw_video = ("--video" in raw_flags) or (os.getenv("TENSORIUM_PLOT_VIDEO", "0") != "0")
raw_show = ("--show" in raw_flags) or (os.getenv("TENSORIUM_PLOT_SHOW", "0") != "0")
raw_single_frame = any(a.startswith("--single-frame-out=") for a in sys.argv[1:])
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib-cache"
if (raw_save_png or raw_video or raw_single_frame) and not raw_show:
    os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, gaussian_filter

data_dir = "Output/viz"
slice_files = sorted(glob.glob(os.path.join(data_dir, "slice_*.csv")))
constraint_files = sorted(glob.glob(os.path.join(data_dir, "constraint_slice_*.csv")))
track_file = os.path.join(data_dir, "puncture_track.csv")
track_df = pd.read_csv(track_file) if os.path.exists(track_file) else None
drift_file = os.path.join(data_dir, "puncture_tracker_drift.csv")
drift_df = pd.read_csv(drift_file) if os.path.exists(drift_file) else None

args = [a for a in sys.argv[1:] if not a.startswith("--")]
flags = {a for a in sys.argv[1:] if a.startswith("--")}

constraint_mode = ("--constraints" in flags) or (
    os.getenv("TENSORIUM_PLOT_CONSTRAINTS", "0") != "0"
)
tracker_drift_mode = ("--tracker-drift" in flags) or (
    os.getenv("TENSORIUM_PLOT_TRACKER_DRIFT", "0") != "0"
)
animate_mode = ("--animate" in flags) or (os.getenv("TENSORIUM_PLOT_ANIMATE", "0") != "0")
save_png_mode = ("--save-png" in flags) or (
    os.getenv("TENSORIUM_PLOT_SAVE_PNG", "0") != "0"
)
video_mode = ("--video" in flags) or (os.getenv("TENSORIUM_PLOT_VIDEO", "0") != "0")
show_mode = ("--show" in flags) or (os.getenv("TENSORIUM_PLOT_SHOW", "0") != "0")
no_smooth = ("--no-smooth" in flags) or (os.getenv("TENSORIUM_PLOT_NO_SMOOTH", "0") != "0")
no_auto_clim = ("--no-auto-clim" in flags) or (
    os.getenv("TENSORIUM_PLOT_NO_AUTO_CLIM", "0") != "0"
)
contours_off = ("--no-contours" in flags) or (
    os.getenv("TENSORIUM_PLOT_NO_CONTOURS", "0") != "0"
)
yt_colors = ("--yt-colors" in flags) or (os.getenv("TENSORIUM_PLOT_USE_YT_COLORS", "1") != "0")
if "--no-yt-colors" in flags:
    yt_colors = False
latex_requested = ("--latex" in flags) or ("--usetex" in flags) or (
    os.getenv("TENSORIUM_PLOT_LATEX", "1") != "0"
)
if ("--no-latex" in flags) or ("--no-usetex" in flags):
    latex_requested = False
smooth_sigma = 0.0 if no_smooth else (0.6 if constraint_mode else 1.0)

_yt_module = None
_yt_checked = False
_yt_warned_missing = False
_yt_warned_profile = False
latex_active = False

if latex_requested:
    has_latex = shutil.which("latex") is not None
    has_tex_backend = (shutil.which("dvipng") is not None) or (shutil.which("dvisvgm") is not None)
    if has_latex and has_tex_backend:
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman"],
                "axes.unicode_minus": False,
                "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
            }
        )
        latex_active = True
    else:
        print(
            "[warn] real LaTeX requested but latex/dvipng is unavailable; falling back to "
            "matplotlib mathtext."
        )

auto_zoom = True
zoom_margin = 15.0
zoom_min_half_width = 19.0

if constraint_mode and tracker_drift_mode:
    print("[ERR] --constraints et --tracker-drift sont exclusifs.")
    sys.exit(1)

files = [] if tracker_drift_mode else (constraint_files if constraint_mode else slice_files)
if tracker_drift_mode:
    if drift_df is None or drift_df.empty:
        print(f"[ERR] fichier manquant ou vide: {drift_file}")
        sys.exit(1)
    if "step" not in drift_df.columns:
        print(f"[ERR] colonne 'step' absente dans {drift_file}")
        sys.exit(1)
    drift_df = drift_df.sort_values("step").drop_duplicates("step", keep="last").reset_index(drop=True)
else:
    if not files:
        wanted = "constraint_slice_*.csv" if constraint_mode else "slice_*.csv"
        print(f"[ERR] aucun fichier {wanted} dans {data_dir}")
        sys.exit(1)

if tracker_drift_mode:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10), constrained_layout=True)
elif constraint_mode:
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    ax1, ax2, ax3, ax4 = axes.ravel()
else:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 6), constrained_layout=True)


def parse_flag_value(name, default):
    prefix = name + "="
    for token in sys.argv[1:]:
        if token.startswith(prefix):
            return token[len(prefix):]
    return default


def parse_int_value(name, default):
    raw = parse_flag_value(name, str(default))
    try:
        return int(raw)
    except ValueError:
        return default


def smooth(M, sigma=1.0):
    return gaussian_filter(M, sigma=sigma)


def extract_step(path):
    name = os.path.basename(path)
    match = re.search(r"(?:slice|constraint_slice|drift_step)_(\d+)\.csv$", name)
    if match:
        return int(match.group(1))
    return None


def parse_requested_step():
    if len(args) == 0:
        return None
    if len(args) > 1:
        print(
            "Usage: python3 plot.py [step] [--animate] [--save-png] [--video] [--fps=N] "
            "[--frames-dir=DIR] [--video-file=FILE.mp4] [--video-codec=auto|libx264|h264_videotoolbox] "
            "[--workers=N] [--dpi=N] "
            "[--constraints] [--tracker-drift] [--no-smooth] "
            "[--no-auto-clim] [--yt-colors|--no-yt-colors] [--latex|--no-latex] "
            "[--no-contours] [--show]"
        )
        sys.exit(1)
    try:
        return int(args[0])
    except ValueError:
        print(f"[ERR] Invalid step '{args[0]}'. Expected an integer.")
        sys.exit(1)


def tex(s_latex, s_plain):
    return s_latex if latex_active else s_plain


def robust_clim(A, lo=2.0, hi=98.0, fallback=(0.0, 1.0)):
    finite = A[np.isfinite(A)]
    if finite.size == 0:
        return fallback
    vmin = float(np.percentile(finite, lo))
    vmax = float(np.percentile(finite, hi))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return fallback
    return vmin, vmax


def robust_symmetric_clim(A, p=99.0, fallback=1.0):
    finite = np.abs(A[np.isfinite(A)])
    if finite.size == 0:
        lim = fallback
    else:
        lim = float(np.percentile(finite, p))
        if not np.isfinite(lim) or lim <= 0.0:
            lim = float(np.max(finite))
        if not np.isfinite(lim) or lim <= 0.0:
            lim = fallback
    return -lim, lim


def robust_positive_floor(A, q=0.5, fallback=1e-14):
    finite = A[np.isfinite(A) & (A > 0.0)]
    if finite.size == 0:
        return fallback
    floor = float(np.percentile(finite, q))
    if not np.isfinite(floor) or floor <= 0.0:
        floor = fallback
    return max(floor, fallback)


def horizon_clim(
    A,
    lo=1.0,
    hi=99.7,
    fallback=(0.0, 1.0),
    vmin_floor=0.0,
    vmax_floor=0.8,
    vmax_cap=1.2,
):
    finite = A[np.isfinite(A)]
    if finite.size == 0:
        return fallback
    vmin = float(np.percentile(finite, lo))
    vmax = float(np.percentile(finite, hi))
    if not np.isfinite(vmin):
        vmin = float(np.min(finite))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = float(np.max(finite))
    if not np.isfinite(vmin):
        vmin = fallback[0]
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = fallback[1]
    vmin = max(vmin, float(vmin_floor))
    vmax = max(vmax, float(vmax_floor))
    vmax = min(vmax, float(vmax_cap))
    if vmax <= vmin:
        vmin = fallback[0]
        vmax = fallback[1]
    # Keep enough dynamic range for visible structure.
    if vmin > 0.80 * vmax:
        vmin = max(fallback[0], 0.0)
    if vmax <= vmin:
        vmax = fallback[1]
    return vmin, vmax


def get_yt_module():
    global _yt_module, _yt_checked, _yt_warned_missing
    if _yt_checked:
        return _yt_module
    _yt_checked = True
    if not yt_colors:
        return None
    try:
        import yt  # type: ignore

        _yt_module = yt
        if os.getenv("TENSORIUM_PLOT_YT_QUIET", "1") != "0":
            try:
                yt.funcs.mylog.setLevel("ERROR")
            except Exception:
                pass
    except Exception as exc:
        _yt_module = None
        if not _yt_warned_missing:
            print(f"[warn] yt colors requested but yt is unavailable ({exc}); fallback to numpy.")
            _yt_warned_missing = True
    return _yt_module


def profile_quantiles(A, field_name, quantiles):
    finite = A[np.isfinite(A)]
    if finite.size == 0:
        return None, "none"

    q = np.clip(np.asarray(quantiles, dtype=float), 0.0, 1.0)
    yt = get_yt_module()
    if yt is not None:
        try:
            arr2 = np.asarray(A, dtype=np.float64)
            ny, nx = arr2.shape
            # yt uniform grid expects x,y,z ordering.
            arr3 = np.ascontiguousarray(arr2.T[:, :, np.newaxis])
            data = {
                ("gas", field_name): (arr3, "dimensionless"),
            }
            bbox = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float64)
            ds = yt.load_uniform_grid(
                data,
                arr3.shape,
                length_unit=1.0,
                bbox=bbox,
                periodicity=(False, False, False),
                unit_system="cgs",
            )
            ad = ds.all_data()
            fmin = float(np.min(finite))
            fmax = float(np.max(finite))
            if np.isfinite(fmin) and np.isfinite(fmax) and fmax > fmin:
                prof = yt.create_profile(
                    ad,
                    [("gas", field_name)],
                    [("index", "cell_volume")],
                    n_bins=1024,
                    extrema={("gas", field_name): (fmin, fmax)},
                    logs={("gas", field_name): False},
                    weight_field=None,
                    accumulation=False,
                    fractional=False,
                )
                bins = np.asarray(prof.x_bins, dtype=np.float64)
                weights = np.asarray(prof[("index", "cell_volume")], dtype=np.float64)
                weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
                weights = np.maximum(weights, 0.0)
                if bins.size == weights.size + 1 and np.sum(weights) > 0.0:
                    cdf = np.cumsum(weights)
                    cdf /= cdf[-1]
                    centers = 0.5 * (bins[:-1] + bins[1:])
                    return np.interp(q, cdf, centers), "yt"
        except Exception as exc:
            global _yt_warned_profile
            if not _yt_warned_profile:
                print(f"[warn] yt profile build failed for field '{field_name}' ({exc}); fallback to numpy.")
                _yt_warned_profile = True

    return np.quantile(finite, q), "numpy"


def profiled_clim(A, field_name, vmax_floor, vmax_cap, default_gamma):
    qs, source = profile_quantiles(A, field_name, [0.02, 0.50, 0.995])
    if qs is None:
        return 0.0, max(vmax_floor, 1.0), default_gamma, source

    q02, q50, q995 = [float(v) for v in qs]
    vmin = max(0.0, q02)
    vmax = q995
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = float(np.max(A[np.isfinite(A)]))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = max(vmax_floor, 1.0)

    if not np.isfinite(vmin):
        vmin = 0.0
    vmax = max(vmax, float(vmax_floor))
    vmax = min(vmax, float(vmax_cap))
    vmax = max(vmax, 1e-12)
    if vmin > 0.80 * vmax:
        vmin = 0.0
    if vmin >= vmax:
        vmin = 0.0

    denom = max(vmax - vmin, 1e-12)
    frac = (q50 - vmin) / denom if np.isfinite(q50) else 0.0
    gamma = default_gamma
    if frac < 0.10:
        gamma = max(0.45, default_gamma - 0.18)
    elif frac > 0.45:
        gamma = min(0.98, default_gamma + 0.12)

    return vmin, vmax, gamma, source


def profiled_symmetric_clim(A, field_name, p=0.99, fallback=1.0):
    absA = np.abs(A)
    qs, source = profile_quantiles(absA, field_name, [p])
    if qs is None:
        return -fallback, fallback, source
    lim = float(qs[0])
    if not np.isfinite(lim) or lim <= 0.0:
        finite = absA[np.isfinite(absA)]
        if finite.size > 0:
            lim = float(np.max(finite))
    if not np.isfinite(lim) or lim <= 0.0:
        lim = fallback
    return -lim, lim, source


def puncture_zoom_window(extent, track_slice):
    if track_slice.empty:
        return None

    last = track_slice.iloc[-1]
    x_left = float(last["x_left"])
    y_left = float(last["y_left"])
    x_right = float(last["x_right"])
    y_right = float(last["y_right"])

    if not np.isfinite([x_left, y_left, x_right, y_right]).all():
        return None

    cx = 0.5 * (x_left + x_right)
    cy = 0.5 * (y_left + y_right)
    sep = np.hypot(x_right - x_left, y_right - y_left)

    half = max(0.5 * sep + zoom_margin, zoom_min_half_width)
    half = min(half, 0.5 * (extent[1] - extent[0]), 0.5 * (extent[3] - extent[2]))

    x_min = max(cx - half, extent[0])
    x_max = min(cx + half, extent[1])
    y_min = max(cy - half, extent[2])
    y_max = min(cy + half, extent[3])
    return x_min, x_max, y_min, y_max


def overlay_track(ax, track_slice):
    if track_slice is None or track_slice.empty:
        return
    ax.plot(
        track_slice["x_left"],
        track_slice["y_left"],
        color="deepskyblue",
        linewidth=1.0,
        alpha=0.85,
    )
    ax.plot(
        track_slice["x_right"],
        track_slice["y_right"],
        color="orange",
        linewidth=1.0,
        alpha=0.85,
    )
    ax.scatter(track_slice["x_left"].iloc[-1], track_slice["y_left"].iloc[-1], color="cyan", s=20)
    ax.scatter(
        track_slice["x_right"].iloc[-1], track_slice["y_right"].iloc[-1], color="orange", s=20
    )


def update_regular(frame_idx):
    current_file = files[frame_idx]
    df = pd.read_csv(current_file)
    step = extract_step(current_file)
    if step is None:
        step = frame_idx

    W = df.pivot(index="y", columns="x", values="W").values
    alpha = df.pivot(index="y", columns="x", values="alpha").values
    mask = df.pivot(index="y", columns="x", values="mask").values

    if smooth_sigma > 0.0:
        W_plot = smooth(W, smooth_sigma)
        alpha_plot = smooth(alpha, smooth_sigma)
    else:
        W_plot = W
        alpha_plot = alpha

    # Physical display clamp: avoid negative undershoot from smoothing/interpolation.
    W_disp = np.maximum(W_plot, 0.0)
    alpha_disp = np.maximum(alpha_plot, 0.0)

    # Keep a distinct black region on BH interiors/horizons.
    # Prefer exported mask, then reinforce with an alpha threshold.
    horizon_mask = np.isfinite(mask) & (mask > 0.5)
    horizon_mask = horizon_mask | (np.isfinite(alpha) & (alpha <= 0.14))
    if np.any(horizon_mask):
        horizon_mask = binary_dilation(horizon_mask, iterations=1)
    horizon_overlay_alpha = 0.94 * horizon_mask.astype(float)

    extent = [df["x"].min(), df["x"].max(), df["y"].min(), df["y"].max()]
    x_unique = np.sort(df["x"].unique())
    y_unique = np.sort(df["y"].unique())
    Xg, Yg = np.meshgrid(x_unique, y_unique)

    ax1.clear()
    ax2.clear()
    ax3.clear()

    w_source = "fixed"
    a_source = "fixed"
    if no_auto_clim:
        w_vmin, w_vmax = 0.0, 1.0
        a_vmin, a_vmax = 0.0, 1.0
        w_gamma, a_gamma = 0.65, 0.75
    else:
        if yt_colors:
            w_vmin, w_vmax, w_gamma, w_source = profiled_clim(
                W_disp, "W", vmax_floor=0.80, vmax_cap=1.30, default_gamma=0.68
            )
            a_vmin, a_vmax, a_gamma, a_source = profiled_clim(
                alpha_disp, "alpha", vmax_floor=0.70, vmax_cap=1.15, default_gamma=0.78
            )
        else:
            w_vmin, w_vmax = horizon_clim(
                W_disp, lo=1.0, hi=99.7, fallback=(0.0, 1.0), vmin_floor=0.0, vmax_floor=0.8, vmax_cap=1.2
            )
            a_vmin, a_vmax = horizon_clim(
                alpha_disp, lo=1.0, hi=99.7, fallback=(0.0, 1.0), vmin_floor=0.0, vmax_floor=0.7, vmax_cap=1.1
            )
            w_gamma, a_gamma = 0.55, 0.62
            w_source = "numpy"
            a_source = "numpy"

    # Nonlinear norm keeps low-but-physical values visible; black remains near horizon interior.
    w_norm = colors.PowerNorm(gamma=w_gamma, vmin=w_vmin, vmax=w_vmax)
    a_norm = colors.PowerNorm(gamma=a_gamma, vmin=a_vmin, vmax=a_vmax)

    ax1.imshow(
        W_disp,
        extent=extent,
        origin="lower",
        cmap="turbo",
        norm=w_norm,
        interpolation="bilinear",
    )
    ax1.imshow(
        np.zeros_like(W_disp),
        extent=extent,
        origin="lower",
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        alpha=horizon_overlay_alpha,
        interpolation="nearest",
    )
    ax1.set_title(
        tex(
            rf"$W\ \left(\mathrm{{step}}={step},\ \mathrm{{profile}}={w_source}\right)$",
            f"W (Conformal Factor)  step = {step}  [profile:{w_source}]",
        )
    )
    ax1.set_aspect("equal")
    ax1.set_xlabel(tex(r"$x$", "x"))
    ax1.set_ylabel(tex(r"$y$", "y"))
    ax1.grid(True, color="white", linestyle="--", linewidth=0.4, alpha=0.35)
    if not contours_off:
        finite_w = W[np.isfinite(W)]
        if finite_w.size > 0:
            w_q = np.percentile(finite_w, [5.0, 15.0, 30.0])
            ax1.contour(Xg, Yg, W, levels=np.unique(w_q), colors="white", linewidths=0.55, alpha=0.6)

    ax2.imshow(
        alpha_disp,
        extent=extent,
        origin="lower",
        cmap="magma",
        norm=a_norm,
        interpolation="bilinear",
    )
    ax2.imshow(
        np.zeros_like(alpha_disp),
        extent=extent,
        origin="lower",
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        alpha=horizon_overlay_alpha,
        interpolation="nearest",
    )
    ax2.set_title(
        tex(
            rf"$\alpha\ \left(\mathrm{{profile}}={a_source}\right)$",
            f"Lapse α  [profile:{a_source}]",
        )
    )
    ax2.set_aspect("equal")
    ax2.set_xlabel(tex(r"$x$", "x"))
    ax2.set_ylabel(tex(r"$y$", "y"))
    ax2.grid(True, color="white", linestyle="--", linewidth=0.4, alpha=0.35)
    if not contours_off:
        finite_a = alpha[np.isfinite(alpha)]
        if finite_a.size > 0:
            a_q = np.percentile(finite_a, [5.0, 15.0, 30.0])
            ax2.contour(Xg, Yg, alpha, levels=np.unique(a_q), colors="cyan", linewidths=0.55, alpha=0.55)

    zoom_window = None
    track_slice = None
    if track_df is not None and not track_df.empty:
        track_slice = track_df[track_df["step"] <= step]
        if not track_slice.empty:
            ax3.plot(
                track_slice["x_left"],
                track_slice["y_left"],
                color="tab:cyan",
                linewidth=1.6,
                label=tex(r"$\mathcal{P}_{\mathrm{L}}$", "Puncture left"),
            )
            ax3.plot(
                track_slice["x_right"],
                track_slice["y_right"],
                color="tab:orange",
                linewidth=1.6,
                label=tex(r"$\mathcal{P}_{\mathrm{R}}$", "Puncture right"),
            )
            ax3.scatter(track_slice["x_left"].iloc[-1], track_slice["y_left"].iloc[-1], color="tab:cyan", s=28)
            ax3.scatter(
                track_slice["x_right"].iloc[-1],
                track_slice["y_right"].iloc[-1],
                color="tab:orange",
                s=28,
            )
            ax3.legend(loc="upper right")
            if auto_zoom:
                zoom_window = puncture_zoom_window(extent, track_slice)
    else:
        ax3.text(
            0.5,
            0.5,
            tex(r"$\mathrm{puncture\_track.csv\ absent}$", "puncture_track.csv absent"),
            transform=ax3.transAxes,
            ha="center",
            va="center",
        )

    if zoom_window is not None:
        x_min, x_max, y_min, y_max = zoom_window
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)
        ax2.set_xlim(x_min, x_max)
        ax2.set_ylim(y_min, y_max)
        ax3.set_xlim(x_min, x_max)
        ax3.set_ylim(y_min, y_max)
    else:
        ax3.set_xlim(extent[0], extent[1])
        ax3.set_ylim(extent[2], extent[3])

    ax3.set_title(tex(r"$\mathrm{Puncture\ Trajectories}$", "Puncture Trajectories"))
    ax3.set_aspect("equal")
    ax3.set_xlabel(tex(r"$x$", "x"))
    ax3.set_ylabel(tex(r"$y$", "y"))
    ax3.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    return []


def update_constraints(frame_idx):
    current_file = files[frame_idx]
    df = pd.read_csv(current_file)
    step = extract_step(current_file)
    if step is None:
        step = frame_idx

    H = df.pivot(index="y", columns="x", values="H").values
    Mnorm = df.pivot(index="y", columns="x", values="Mnorm").values
    Cnorm = df.pivot(index="y", columns="x", values="Cnorm").values
    if smooth_sigma > 0.0:
        H_plot = smooth(H, smooth_sigma)
        Mnorm_plot = smooth(Mnorm, smooth_sigma)
        Cnorm_plot = smooth(Cnorm, smooth_sigma)
    else:
        H_plot = H
        Mnorm_plot = Mnorm
        Cnorm_plot = Cnorm

    absH = np.abs(H_plot)
    eps_H = robust_positive_floor(absH, q=0.5, fallback=1e-14)
    eps_M = robust_positive_floor(Mnorm_plot, q=0.5, fallback=1e-14)
    eps_C = robust_positive_floor(Cnorm_plot, q=0.5, fallback=1e-14)
    log_absH = np.log10(np.maximum(absH, eps_H))
    log_M = np.log10(np.maximum(Mnorm_plot, eps_M))
    log_C = np.log10(np.maximum(Cnorm_plot, eps_C))

    extent = [df["x"].min(), df["x"].max(), df["y"].min(), df["y"].max()]
    x_unique = np.sort(df["x"].unique())
    y_unique = np.sort(df["y"].unique())
    Xg, Yg = np.meshgrid(x_unique, y_unique)

    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()

    if no_auto_clim:
        h_vmin, h_vmax = -1.0, 1.0
        hlog_vmin, hlog_vmax = -10.0, 0.0
        mlog_vmin, mlog_vmax = -10.0, 0.0
        clog_vmin, clog_vmax = -10.0, 0.0
    else:
        if yt_colors:
            h_vmin, h_vmax, _ = profiled_symmetric_clim(H_plot, "H_abs", p=0.99, fallback=1.0)

            def profile_log_bounds(A, field_name):
                qs, _ = profile_quantiles(A, field_name, [0.02, 0.98])
                if qs is None:
                    return robust_clim(A, lo=2.0, hi=98.0, fallback=(-12.0, -1.0))
                lo_v, hi_v = float(qs[0]), float(qs[1])
                if not np.isfinite(lo_v) or not np.isfinite(hi_v) or hi_v <= lo_v:
                    return robust_clim(A, lo=2.0, hi=98.0, fallback=(-12.0, -1.0))
                return lo_v, hi_v

            hlog_vmin, hlog_vmax = profile_log_bounds(log_absH, "log_absH")
            mlog_vmin, mlog_vmax = profile_log_bounds(log_M, "log_M")
            clog_vmin, clog_vmax = profile_log_bounds(log_C, "log_C")
        else:
            h_vmin, h_vmax = robust_symmetric_clim(H_plot, p=99.0, fallback=1.0)
            hlog_vmin, hlog_vmax = robust_clim(log_absH, lo=2.0, hi=98.0, fallback=(-12.0, -1.0))
            mlog_vmin, mlog_vmax = robust_clim(log_M, lo=2.0, hi=98.0, fallback=(-12.0, -1.0))
            clog_vmin, clog_vmax = robust_clim(log_C, lo=2.0, hi=98.0, fallback=(-12.0, -1.0))

    ax1.imshow(
        H_plot,
        extent=extent,
        origin="lower",
        cmap="magma",
        vmin=h_vmin,
        vmax=h_vmax,
        interpolation="bicubic",
    )
    ax1.set_title(
        tex(
            rf"$\mathcal{{H}}\ \left(\mathrm{{signed}},\ \mathrm{{step}}={step}\right)$",
            f"Hamiltonian H (signed)  step = {step}",
        )
    )
    ax1.set_aspect("equal")
    ax1.set_xlabel(tex(r"$x$", "x"))
    ax1.set_ylabel(tex(r"$y$", "y"))
    ax1.grid(True, color="white", linestyle="--", linewidth=0.35, alpha=0.3)
    if not contours_off:
        levels = np.linspace(h_vmin, h_vmax, 9)
        ax1.contour(Xg, Yg, H, levels=levels, colors="black", linewidths=0.35, alpha=0.25)

    ax2.imshow(
        log_absH,
        extent=extent,
        origin="lower",
        cmap="viridis",
        vmin=hlog_vmin,
        vmax=hlog_vmax,
        interpolation="bicubic",
    )
    ax2.set_title(tex(r"$\log_{10}\!\left|\mathcal{H}\right|$", "log10(|H|)"))
    ax2.set_aspect("equal")
    ax2.set_xlabel(tex(r"$x$", "x"))
    ax2.set_ylabel(tex(r"$y$", "y"))
    ax2.grid(True, color="white", linestyle="--", linewidth=0.35, alpha=0.3)
    if not contours_off:
        levels = np.linspace(hlog_vmin, hlog_vmax, 9)
        ax2.contour(Xg, Yg, log_absH, levels=levels, colors="white", linewidths=0.35, alpha=0.35)

    ax3.imshow(
        log_M,
        extent=extent,
        origin="lower",
        cmap="viridis",
        vmin=mlog_vmin,
        vmax=mlog_vmax,
        interpolation="bicubic",
    )
    ax3.set_title(tex(r"$\log_{10}\!\left\|\mathcal{M}\right\|$", "log10(|M|)"))
    ax3.set_aspect("equal")
    ax3.set_xlabel(tex(r"$x$", "x"))
    ax3.set_ylabel(tex(r"$y$", "y"))
    ax3.grid(True, color="white", linestyle="--", linewidth=0.35, alpha=0.3)
    if not contours_off:
        levels = np.linspace(mlog_vmin, mlog_vmax, 9)
        ax3.contour(Xg, Yg, log_M, levels=levels, colors="white", linewidths=0.35, alpha=0.35)

    ax4.imshow(
        log_C,
        extent=extent,
        origin="lower",
        cmap="viridis",
        vmin=clog_vmin,
        vmax=clog_vmax,
        interpolation="bicubic",
    )
    ax4.set_title(
        tex(
            r"$\log_{10}\!\left\|\mathcal{C}_{\Gamma}\right\|$",
            "log10(|C|)  (Gamma constraint)",
        )
    )
    ax4.set_aspect("equal")
    ax4.set_xlabel(tex(r"$x$", "x"))
    ax4.set_ylabel(tex(r"$y$", "y"))
    ax4.grid(True, color="white", linestyle="--", linewidth=0.35, alpha=0.3)
    if not contours_off:
        levels = np.linspace(clog_vmin, clog_vmax, 9)
        ax4.contour(Xg, Yg, log_C, levels=levels, colors="white", linewidths=0.35, alpha=0.35)

    zoom_window = None
    track_slice = None
    if track_df is not None and not track_df.empty:
        track_slice = track_df[track_df["step"] <= step]
        if auto_zoom and not track_slice.empty:
            zoom_window = puncture_zoom_window(extent, track_slice)

    for ax in (ax1, ax2, ax3, ax4):
        if track_slice is not None and not track_slice.empty:
            overlay_track(ax, track_slice)
        if zoom_window is not None:
            x_min, x_max, y_min, y_max = zoom_window
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
    return []


def update_tracker_drift(frame_idx):
    row = drift_df.iloc[frame_idx]
    step = int(row["step"])
    hist = drift_df.iloc[: frame_idx + 1]

    ax1.clear()
    ax2.clear()

    x = hist["t"].to_numpy(dtype=float) if "t" in hist.columns else hist["step"].to_numpy(dtype=float)
    d_left = np.clip(hist["drift_left"].to_numpy(dtype=float), 1e-16, None)
    d_right = np.clip(hist["drift_right"].to_numpy(dtype=float), 1e-16, None)

    ax1.semilogy(x, d_left, color="tab:cyan", linewidth=1.5, label=tex(r"$d_{\mathrm{L}}$", "drift_left"))
    ax1.semilogy(x, d_right, color="tab:orange", linewidth=1.5, label=tex(r"$d_{\mathrm{R}}$", "drift_right"))
    if "recentered" in hist.columns:
        rec = hist[hist["recentered"] != 0]
        if not rec.empty:
            x_rec = rec["t"].to_numpy(dtype=float) if "t" in rec.columns else rec["step"].to_numpy(dtype=float)
            ax1.scatter(
                x_rec,
                np.clip(rec["drift_left"].to_numpy(dtype=float), 1e-16, None),
                marker="x",
                s=20,
                color="red",
                label=tex(r"$\mathrm{recenter}$", "recenter"),
            )
    ax1.set_title(
        tex(
            rf"$\mathrm{{Puncture\ Tracker\ Drift}}\ \left(\mathrm{{step}}={step}\right)$",
            f"Puncture Tracker Drift (step={step})",
        )
    )
    ax1.set_xlabel(tex(r"$t$", "t") if "t" in hist.columns else tex(r"$n_{\mathrm{step}}$", "step"))
    ax1.set_ylabel(tex(r"$d\ [\mathrm{cells}]$", "drift (cells)"))
    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax1.legend(loc="upper right")

    ax2.plot(
        hist["shift_x_left"],
        hist["shift_y_left"],
        color="tab:cyan",
        linewidth=1.6,
        label=tex(r"$\beta\mathrm{-tracker}_{\mathrm{L}}$", "shift left"),
    )
    ax2.plot(
        hist["shift_x_right"],
        hist["shift_y_right"],
        color="tab:orange",
        linewidth=1.6,
        label=tex(r"$\beta\mathrm{-tracker}_{\mathrm{R}}$", "shift right"),
    )
    ax2.plot(
        hist["min_x_left"],
        hist["min_y_left"],
        color="tab:cyan",
        linestyle="--",
        linewidth=1.2,
        alpha=0.85,
        label=tex(r"$\chi\mathrm{-min}_{\mathrm{L}}$", "min left"),
    )
    ax2.plot(
        hist["min_x_right"],
        hist["min_y_right"],
        color="tab:orange",
        linestyle="--",
        linewidth=1.2,
        alpha=0.85,
        label=tex(r"$\chi\mathrm{-min}_{\mathrm{R}}$", "min right"),
    )
    ax2.scatter(hist["shift_x_left"].iloc[-1], hist["shift_y_left"].iloc[-1], color="tab:cyan", s=25)
    ax2.scatter(hist["shift_x_right"].iloc[-1], hist["shift_y_right"].iloc[-1], color="tab:orange", s=25)
    ax2.scatter(
        hist["min_x_left"].iloc[-1],
        hist["min_y_left"].iloc[-1],
        color="tab:cyan",
        s=22,
        marker="x",
    )
    ax2.scatter(
        hist["min_x_right"].iloc[-1],
        hist["min_y_right"].iloc[-1],
        color="tab:orange",
        s=22,
        marker="x",
    )

    x_all = np.concatenate(
        [
            hist["shift_x_left"].to_numpy(dtype=float),
            hist["shift_x_right"].to_numpy(dtype=float),
            hist["min_x_left"].to_numpy(dtype=float),
            hist["min_x_right"].to_numpy(dtype=float),
        ]
    )
    y_all = np.concatenate(
        [
            hist["shift_y_left"].to_numpy(dtype=float),
            hist["shift_y_right"].to_numpy(dtype=float),
            hist["min_y_left"].to_numpy(dtype=float),
            hist["min_y_right"].to_numpy(dtype=float),
        ]
    )
    finite = np.isfinite(x_all) & np.isfinite(y_all)
    if np.any(finite):
        x_ok = x_all[finite]
        y_ok = y_all[finite]
        pad_x = max(0.05 * (float(np.max(x_ok)) - float(np.min(x_ok))), 0.2)
        pad_y = max(0.05 * (float(np.max(y_ok)) - float(np.min(y_ok))), 0.2)
        ax2.set_xlim(float(np.min(x_ok)) - pad_x, float(np.max(x_ok)) + pad_x)
        ax2.set_ylim(float(np.min(y_ok)) - pad_y, float(np.max(y_ok)) + pad_y)
    ax2.set_title(
        tex(
            r"$\beta\mathrm{-integrated\ tracker}\ \mathrm{vs}\ \chi\mathrm{-min\ tracker}$",
            "Shift-Integrated Tracker vs Minima Tracker",
        )
    )
    ax2.set_aspect("equal")
    ax2.set_xlabel(tex(r"$x$", "x"))
    ax2.set_ylabel(tex(r"$y$", "y"))
    ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax2.legend(loc="upper right")
    return []


def update(frame_idx):
    if tracker_drift_mode:
        return update_tracker_drift(frame_idx)
    if constraint_mode:
        return update_constraints(frame_idx)
    return update_regular(frame_idx)


def resolve_step_index(requested_step, step_to_index):
    if requested_step is None:
        return len(files) - 1
    if requested_step not in step_to_index:
        available_steps = sorted(step_to_index.keys())
        if tracker_drift_mode:
            kind = "tracker_drift"
        else:
            kind = "constraint_slice" if constraint_mode else "slice"
        print(f"[ERR] {kind} step={requested_step} introuvable.")
        if available_steps:
            print(
                f"Steps disponibles: {available_steps[0]} .. {available_steps[-1]} "
                f"(n={len(available_steps)})"
            )
        sys.exit(1)
    return step_to_index[requested_step]


def export_frames_to_png(frame_indices, out_dir, dpi):
    os.makedirs(out_dir, exist_ok=True)
    for stale in glob.glob(os.path.join(out_dir, "frame_*.png")):
        os.remove(stale)

    total = len(frame_indices)
    written = [os.path.join(out_dir, f"frame_{n:05d}.png") for n in range(total)]
    return written


def render_single_frame_to_png(frame_idx, out_path, dpi):
    update(frame_idx)
    out_parent = os.path.dirname(out_path)
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)


def build_child_render_command(step, out_path, dpi):
    cmd = [sys.executable, os.path.abspath(__file__), str(step), f"--single-frame-out={out_path}", f"--dpi={dpi}"]
    if tracker_drift_mode:
        cmd.append("--tracker-drift")
    if constraint_mode:
        cmd.append("--constraints")
    if no_smooth:
        cmd.append("--no-smooth")
    if no_auto_clim:
        cmd.append("--no-auto-clim")
    if yt_colors:
        cmd.append("--yt-colors")
    else:
        cmd.append("--no-yt-colors")
    if latex_active:
        cmd.append("--latex")
    else:
        cmd.append("--no-latex")
    if contours_off:
        cmd.append("--no-contours")
    return cmd


def export_frames_to_png_parallel(frame_indices, out_dir, dpi, workers):
    written = export_frames_to_png(frame_indices, out_dir, dpi)
    total = len(frame_indices)
    workers = max(1, workers)

    if workers == 1:
        for n, frame_idx in enumerate(frame_indices):
            out_path = written[n]
            render_single_frame_to_png(frame_idx, out_path, dpi)
            if (n + 1) % 25 == 0 or (n + 1) == total:
                print(f"[png] {n + 1}/{total} -> {out_path}")
        return written

    def run_one(n, frame_idx):
        if tracker_drift_mode:
            step = int(drift_df.iloc[frame_idx]["step"])
        else:
            step = extract_step(files[frame_idx])
            if step is None:
                step = frame_idx
        out_path = written[n]
        cmd = build_child_render_command(step, out_path, dpi)
        # Isolate matplotlib/TeX caches per child render to avoid concurrent cache corruption
        # when many workers render LaTeX text at the same time.
        child_env = os.environ.copy()
        child_cache = os.path.join("/tmp", "matplotlib-cache-workers", f"frame_{n:05d}")
        os.makedirs(child_cache, exist_ok=True)
        child_env["MPLCONFIGDIR"] = child_cache
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=child_env,
        )
        return n, out_path, proc.returncode, proc.stdout

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_one, n, frame_idx) for n, frame_idx in enumerate(frame_indices)]
        for fut in as_completed(futs):
            n, out_path, rc, out = fut.result()
            done += 1
            if rc != 0:
                print(f"[ERR] frame {n} failed (step render): {out_path}")
                print(out)
                raise RuntimeError(f"frame render failed for {out_path}")
            if done % 25 == 0 or done == total:
                print(f"[png] {done}/{total} -> {out_path}")
    return written


def choose_video_codec(ffmpeg, requested_codec):
    req = (requested_codec or "auto").strip()
    if req != "auto":
        return req

    enc = subprocess.run(
        [ffmpeg, "-hide_banner", "-encoders"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    out = enc.stdout if enc.returncode == 0 else ""
    if "h264_videotoolbox" in out:
        return "h264_videotoolbox"
    return "libx264"


def build_video_with_ffmpeg(frames_dir, out_file, fps, requested_codec="auto"):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("[ERR] ffmpeg introuvable. Installe ffmpeg pour exporter la video.")
        return False

    out_parent = os.path.dirname(out_file)
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)

    def run_encode(codec):
        cmd = [
            ffmpeg,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            os.path.join(frames_dir, "frame_%05d.png"),
            "-c:v",
            codec,
        ]
        if codec == "h264_videotoolbox":
            cmd += ["-allow_sw", "1", "-b:v", "10M"]
        cmd += [
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            out_file,
        ]
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    codec = choose_video_codec(ffmpeg, requested_codec)
    proc = run_encode(codec)
    if proc.returncode == 0:
        print(f"[OK ] video: {out_file} (codec={codec})")
        return True

    # Robust fallback: if GPU codec fails, retry with libx264.
    if codec != "libx264":
        print(f"[warn] codec {codec} a echoue, fallback libx264")
        proc2 = run_encode("libx264")
        if proc2.returncode == 0:
            print(f"[OK ] video: {out_file} (codec=libx264)")
            return True
        print("[ERR] ffmpeg a echoue (codec primaire + fallback):")
        print(proc.stdout)
        print(proc2.stdout)
        return False

    print("[ERR] ffmpeg a echoue:")
    print(proc.stdout)
    return False


requested_step = parse_requested_step()
if tracker_drift_mode:
    tracker_steps = drift_df["step"].astype(int).tolist()
    files = [f"drift_step_{s}.csv" for s in tracker_steps]
    step_to_index = {int(step): idx for idx, step in enumerate(tracker_steps)}
else:
    step_to_index = {}
    for idx, path in enumerate(files):
        step = extract_step(path)
        if step is not None:
            step_to_index[step] = idx

target_idx = resolve_step_index(requested_step, step_to_index)
full_or_prefix_frames = list(range(target_idx + 1))

if tracker_drift_mode:
    mode_tag = "tracker_drift"
elif constraint_mode:
    mode_tag = "constraints"
else:
    mode_tag = "fields"
default_frames_dir = os.path.join(data_dir, f"{mode_tag}_frames")
default_video_file = os.path.join(data_dir, f"{mode_tag}.mp4")
frames_dir = parse_flag_value("--frames-dir", os.getenv("TENSORIUM_PLOT_FRAMES_DIR", default_frames_dir))
video_file = parse_flag_value("--video-file", os.getenv("TENSORIUM_PLOT_VIDEO_FILE", default_video_file))
video_codec = parse_flag_value("--video-codec", os.getenv("TENSORIUM_PLOT_VIDEO_CODEC", "auto"))
fps = max(1, parse_int_value("--fps", int(os.getenv("TENSORIUM_PLOT_FPS", "24"))))
png_dpi = max(72, parse_int_value("--dpi", int(os.getenv("TENSORIUM_PLOT_DPI", "170"))))
workers = max(1, parse_int_value("--workers", int(os.getenv("TENSORIUM_PLOT_WORKERS", "1"))))
single_frame_out = parse_flag_value("--single-frame-out", "")

if single_frame_out:
    update(target_idx)
    out_parent = os.path.dirname(single_frame_out)
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)
    fig.savefig(single_frame_out, dpi=png_dpi)
    print(f"[OK ] single frame: {single_frame_out}")
    plt.close(fig)
    sys.exit(0)

if save_png_mode or video_mode:
    if requested_step is None:
        export_frame_indices = list(range(len(files)))
    else:
        export_frame_indices = full_or_prefix_frames if (animate_mode or video_mode) else [target_idx]

    written = export_frames_to_png_parallel(export_frame_indices, frames_dir, png_dpi, workers)
    print(f"[OK ] PNG export: {len(written)} frames in {frames_dir}")
    if video_mode:
        if len(written) < 2:
            print("[warn] video non creee: il faut au moins 2 frames.")
        else:
            build_video_with_ffmpeg(frames_dir, video_file, fps, video_codec)

ani = None
if animate_mode:
    ani = animation.FuncAnimation(fig, update, frames=full_or_prefix_frames, interval=40, blit=False)
else:
    update(target_idx)

if show_mode or not (save_png_mode or video_mode):
    plt.show()
else:
    plt.close(fig)
