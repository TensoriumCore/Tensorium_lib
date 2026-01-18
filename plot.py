import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
import os
import numpy as np
from scipy.ndimage import gaussian_filter

data_dir = "Output/viz"
files = sorted(glob.glob(os.path.join(data_dir, "slice_*.csv")))

if not files:
    exit()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)


def smooth(M, sigma=1.0):
    return gaussian_filter(M, sigma=sigma)


def update(frame_idx):
    df = pd.read_csv(files[frame_idx])

    W = df.pivot(index='y', columns='x', values='W').values
    alpha = df.pivot(index='y', columns='x', values='alpha').values
    mask = df.pivot(index='y', columns='x', values='mask').values

    W = smooth(W, 1.0)
    alpha = smooth(alpha, 1.0)

    extent = [df['x'].min(), df['x'].max(), df['y'].min(), df['y'].max()]

    ax1.clear()
    ax2.clear()

    im1 = ax1.imshow(
        W, extent=extent, origin='lower',
        cmap='cividis', vmin=0.0, vmax=0.8,
        interpolation='bicubic'
    )

    ax1.imshow(
        mask, extent=extent, origin='lower',
        cmap='gray', alpha=0.25,
        interpolation='nearest'
    )

    ax1.set_title(f"W (Conformal Factor)  t = {frame_idx}")
    ax1.set_aspect('equal')
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    im2 = ax2.imshow(
        alpha, extent=extent, origin='lower',
        cmap='inferno', vmin=0.0, vmax=0.8,
        interpolation='bicubic'
    )

    ax2.set_title("Lapse α")
    ax2.set_aspect('equal')
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")

    return im1, im2


ani = animation.FuncAnimation(
    fig, update, frames=len(files), interval=80, blit=False)
plt.show()
