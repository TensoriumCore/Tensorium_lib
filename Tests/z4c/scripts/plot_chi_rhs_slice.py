import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fname = sys.argv[1] if len(sys.argv) > 1 else "chi_rhs_slice.csv"
print(f"[plot] loading {fname}")
df = pd.read_csv(fname)
x_vals = np.sort(df['x'].unique())
z_vals = np.sort(df['z'].unique())
NX = len(x_vals)
NZ = len(z_vals)

def reshape(col):
    return df[col].values.reshape((NX, NZ))

alpha = reshape('alpha')
chi = reshape('chi')
rhs = reshape('rhs_chi')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
labels = [r"$\alpha$", r"$\chi$", r"$|\mathrm{RHS}_\chi|$"]
for ax, data, title in zip(axes, [alpha, chi, np.abs(rhs)], labels):
    im = ax.imshow(data, origin='lower', extent=[z_vals.min(), z_vals.max(), x_vals.min(), x_vals.max()],
                   aspect='auto', cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel('z')
    ax.set_ylabel('x')
    fig.colorbar(im, ax=ax)
plt.tight_layout()
plt.show()
