import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("grid_structure.csv")
NX_TOT = df['i'].max() + 1
NY_TOT = df['j'].max() + 1

chi = df['chi'].values.reshape((NX_TOT, NY_TOT))
zone = df['zone_type'].values.reshape((NX_TOT, NY_TOT))

plt.figure(figsize=(10, 8))

plt.imshow(chi, origin='lower', cmap='viridis', extent=[0, NX_TOT, 0, NY_TOT])
plt.colorbar(label='chi')

ng = 3
plt.axvline(x=ng, color='red', linestyle='--',
            alpha=0.8, label='Frontière Ghost')
plt.axvline(x=NX_TOT-ng, color='red', linestyle='--')
plt.axhline(y=ng, color='red', linestyle='--')
plt.axhline(y=NY_TOT-ng, color='red', linestyle='--')

plt.title("Visualisation de la structure mémoire (Intérieur + Halos)")
plt.xlabel("Index mémoire i")
plt.ylabel("Index mémoire j")
plt.legend()
plt.show()
