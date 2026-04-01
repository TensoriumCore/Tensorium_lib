import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("bowen_york_constraints.csv")
r = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(r, np.abs(df['H']), s=5, label='|H|')
ax.scatter(r, df['M'], s=5, label='|M|')
ax.scatter(r, df['C'], s=5, label='|C|')
ax.set_yscale('log')
ax.set_xlabel('r')
ax.set_ylabel('Constraint magnitude')
ax.legend()
ax.set_title('Bowen-York constraints vs radius')
plt.tight_layout()
plt.show()
