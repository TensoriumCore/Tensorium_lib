import sys
import time

sys.path.append("pybuild")
from morpheus import *

matA = Matrix(2, 3)
matA.fill([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

matB = Matrix(2, 3)
matB.fill([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])

print("matA + matB =")
morph.add_mat(matA, matB).print()

print("matA - matB =")
morph.sub_mat(matA, matB).print()

print("matA * 2.0 =")
morph.scl_mat(matA, 2.0).print()

v = Vector([1.0, 2.0, 3.0])
v2 = Vector([4.0, 5.0, 6.0])

print("v =", v)
print("len(v) =", len(v))
print("v + v2 =", morph.add_vec(v, v2))
print("v - v2 =", morph.sub_vec(v, v2))
print("v * 2.0 =", morph.scl_vec(v, 2.0))
print("dot(v, v2) =", morph.dot_vec(v, v2))
print("norm_1(v) =", morph.norm_1(v))
print("norm_2(v) =", morph.norm_2(v))
print("norm_inf(v) =", morph.norm_inf(v))
print("cosine(v, v2) =", morph.cosine(v, v2))
print("lerp(v, v2, 0.5) =", morph.lerp(v, v2, 0.5))
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
