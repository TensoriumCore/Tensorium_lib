import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("benchmark_results.csv")

df_float = df[df["Type"] == "float"]
df_double = df[df["Type"] == "double"]

def plot_performance(df, dtype):
    plt.figure(figsize=(10, 6))
    plt.plot(df["N"], df["GFLOPS_Custom"], marker="o", label="Custom AVX2")
    plt.plot(df["N"], df["GFLOPS_BLAS"], marker="s", label="BLAS SGEMM")
    plt.title(f"Matrix Multiplication Performance ({dtype})")
    plt.xlabel("Matrix Size (N x N)")
    plt.ylabel("GFLOP/s")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"perf_{dtype}.png")
    plt.show()

def plot_error(df, dtype):
    plt.figure(figsize=(10, 4))
    plt.plot(df["N"], df["MaxAbsError"], marker="x", color="red")
    plt.title(f"Max Absolute Error ({dtype})")
    plt.xlabel("Matrix Size (N x N)")
    plt.ylabel("Max Abs Error")
    plt.yscale("log")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"error_{dtype}.png")
    plt.show()

plot_performance(df_float, "float")
plot_performance(df_double, "double")

plot_error(df_float, "float")
plot_error(df_double, "double")
