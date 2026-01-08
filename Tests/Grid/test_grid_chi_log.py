import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("log_chi_slice.csv")
nx_tot = df['i'].max() + 1
ny_tot = df['j'].max() + 1

log_chi = df['log_chi'].values.reshape((nx_tot, ny_tot))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

im = ax1.imshow(log_chi, origin='lower', cmap='plasma')
ax1.set_title(r'Puncture structure ($\log_{10}(\chi)$)')
ax1.set_xlabel('Index i')
ax1.set_ylabel('Index j')
fig.colorbar(im, ax=ax1, label=r'$\log_{10}(\chi)$')

ng = 3
ax1.axvline(x=ng, color='cyan', linestyle='--', alpha=0.5)
ax1.axvline(x=nx_tot-ng, color='cyan', linestyle='--')
ax1.axhline(y=ng, color='cyan', linestyle='--')
ax1.axhline(y=ny_tot-ng, color='cyan', linestyle='--')

center_idx = ny_tot // 2
ax2.plot(log_chi[center_idx, :], color='red', lw=2)
ax2.set_title('Trumpet slicing')
ax2.set_xlabel('Index i')
ax2.set_ylabel(r'$\log_{10}(\chi)$')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
