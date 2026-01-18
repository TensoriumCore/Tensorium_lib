import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_a = pd.read_csv("alpha_slice.csv")
N = int(np.sqrt(len(df_a)))

alpha = df_a['alpha'].values.reshape((N, N))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

im1 = ax1.imshow(alpha, origin='lower', cmap='magma')
ax1.set_title(r'$\alpha$')
fig.colorbar(im1, ax=ax1)

center_idx = N // 2
ax2.plot(alpha[center_idx, :], label=r'$\alpha$')

ax2.set_title(r'$\alpha$ 1D Center')
ax2.legend()
plt.show()
