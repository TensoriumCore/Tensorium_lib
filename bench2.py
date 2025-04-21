
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("benchmark_results2.csv")
grouped = df.groupby("Size")["GFLOP/s"]

avg_gflops = grouped.mean()
peak_gflops = grouped.max()

plt.plot(avg_gflops.index, avg_gflops.values, '*-', label="Average")
plt.plot(peak_gflops.index, peak_gflops.values, '*-', label="Peak")
plt.xlabel("Matrix size")
plt.ylabel("GFLOP/s")
plt.legend()
plt.title("Morpheus mul_mat() Performance")
plt.show()
