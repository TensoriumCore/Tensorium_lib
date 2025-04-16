import sys
import time

sys.path.append("pybuild")
from morpheus import *


# === Repr
print("repr(matA) =", repr(matA))
print("repr(v) =", repr(v))

# === List available morph functions
print("\n=== morph functions ===")
print(dir(morph))

# === Benchmark
def benchmark_large_matrix(N):
    print(f"\n=== Benchmarking Python × Morpheus (N = {N}) ===")
    A = Matrix(N, N)
    B = Matrix(N, N)

    print("Initializing matrices...")
    start_init = time.perf_counter()
    fill_data = [[1.0] * N for _ in range(N)]
    A.fill(fill_data)
    B.fill(fill_data)
    end_init = time.perf_counter()
    print(f"Matrices initialized in {end_init - start_init:.3f} s")

    start = time.perf_counter()
    C = morph.mul(A, B)
    end = time.perf_counter()

    elapsed = end - start
    gflops = 2 * N**3 / (elapsed * 1e9)
    print(f"Time: {elapsed:.3f} s")
    print(f"Performance: {gflops:.2f} GFLOP/s")
    print(f"Sample result: C[0, 0] = {C[0, 0]} (Expected: {N})")

benchmark_large_matrix(8192)
