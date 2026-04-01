import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BIN_COUNT = 64

if __name__ == "__main__":
    df = pd.read_csv("bowen_york_constraints.csv")
    r = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)

    bins = np.linspace(r.min(), r.max(), BIN_COUNT + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    H_vals = np.zeros_like(centers)
    M_vals = np.zeros_like(centers)
    C_vals = np.zeros_like(centers)

    for idx in range(BIN_COUNT):
        mask = (r >= bins[idx]) & (r < bins[idx + 1])
        if not mask.any():
            H_vals[idx] = np.nan
            M_vals[idx] = np.nan
            C_vals[idx] = np.nan
            continue
        H_vals[idx] = np.nanmean(np.abs(df['H'][mask]))
        M_vals[idx] = np.nanmean(df['M'][mask])
        C_vals[idx] = np.nanmean(df['C'][mask])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(centers, H_vals, label='⟨|H|⟩')
    ax.plot(centers, M_vals, label='⟨|M|⟩')
    ax.plot(centers, C_vals, label='⟨|C|⟩')
    ax.set_yscale('log')
    ax.set_xlabel('r (bin centre)')
    ax.set_ylabel('Mean constraint magnitude')
    ax.set_title('Bowen–York radial constraints (binned)')
    ax.legend()
    ax.grid(True, which='both', ls='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
