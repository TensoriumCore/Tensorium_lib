import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("bowen_york_slice_xy.csv")
x_vals = np.sort(df['x'].unique())
z_vals = np.sort(df['z'].unique())
NX = len(x_vals)
NZ = len(z_vals)

alpha = df['alpha'].values.reshape((NX, NZ))
chi = df['chi'].values.reshape((NX, NZ))
K = df['K'].values.reshape((NX, NZ))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, data, title in zip(axes, [alpha, chi, K], [r"$\alpha$", r"$\chi$", r"$K$"]):
    im = ax.imshow(data, origin='lower', extent=[z_vals.min(), z_vals.max(), x_vals.min(), x_vals.max()],
                   aspect='auto', cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel('z')
    ax.set_ylabel('x')
    fig.colorbar(im, ax=ax)
plt.tight_layout()
plt.show()
