import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 3:
    print("Usage: python plot_gamma_rhs_slice.py <gamma_csv> <rhs_csv> [output.png]")
    sys.exit(1)

gamma_csv, rhs_csv = sys.argv[1:3]
output = sys.argv[3] if len(sys.argv) > 3 else None

def load_slice(path):
    df = pd.read_csv(path)
    xs = np.sort(df['x'].unique())
    ys = np.sort(df['y'].unique())
    NX, NY = len(xs), len(ys)
    data = df.iloc[:, 2].values.reshape((NX, NY))
    return xs, ys, data

xg, yg, gamma = load_slice(gamma_csv)
_, _, rhs = load_slice(rhs_csv)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, data, title in zip(axes, [gamma, rhs], ["gamma_xx slice", "rhs_gamma_xx slice"]):
    im = ax.imshow(data, origin='lower', extent=[yg.min(), yg.max(), xg.min(), xg.max()],
                   aspect='auto', cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel('y')
    ax.set_ylabel('x')
    fig.colorbar(im, ax=ax)
plt.tight_layout()
if output:
    plt.savefig(output)
else:
    plt.show()
