#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Plot CCZ4-prime stability constraints vs time")
parser.add_argument("csv", nargs="?", default="Output/tests/stability_test_ccz4_prime.csv",
                    help="CSV with columns t,L2_H,L2_M produced by the CCZ4-prime stability test")
args = parser.parse_args()

df = pd.read_csv(args.csv)
required = {"t", "L2_H", "L2_M"}
if not required.issubset(df.columns):
    raise SystemExit(f"{args.csv} is missing required columns {required}")

plt.figure(figsize=(8, 5))
plt.plot(df["t"], df["L2_H"], label="L2(H)")
plt.plot(df["t"], df["L2_M"], label="L2(M)")
plt.yscale('log')
plt.xlabel('t')
plt.ylabel('Constraint L2 norm')
plt.title('CCZ4-prime stability test')
plt.legend()
plt.tight_layout()
plt.show()
