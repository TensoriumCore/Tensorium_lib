import sys
import time

sys.path.append("pybuild")
from morpheus import *
import morpheus as morph
from mopheus import Vector, Matrix

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


A = Matrix(3, 3)
A.fill([
    [1.0, 2.0, 3.0],
    [0.0, 1.0, 4.0],
    [5.0, 6.0, 0.0],
])

v = Vector([1.0, 2.0, 3.0])

print("Matrix A:")
print(A)

print("\nAddition A + A:")
print(morph.add_mat(A, A))

print("\nSubtraction A - A:")
print(morph.sub_mat(A, A))

print("\nScaling A * 2:")
print(morph.scl_mat(A, 2.0))

print("\nMultiplication A * A:")
print(morph.mul(A, A))

print("\nTranspose of A:")
print(morph.transpose_mat(A))

print("\nTrace of A:")
print(morph.trace_mat(A))

print("\nMatrix A multiplied by vector v:")
print(morph.mul_vec(A, v))

print("\nInverse of A:")
print(morph.inverse_mat(A))

print("\nDeterminant of A:")
print(morph.det_mat(A))

print("\nRank of A:")
print(morph.rank_mat(A))
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



def test_solver():
    A = Matrix(3, 3)
    B = Matrix(3, 3)

    A.fill([
        [2.0, 1.0, -1.0],
        [-3.0, -1.0, 2.0],
        [-2.0, 1.0, 2.0],
    ])
    
    B.fill([
        [10.0, 1.0, 1.0],
        [2.0, 10.0, 1.0],
        [2.0, 2.0, 10.0],
    ])

    b = Vector([8.0, -11.0, -3.0])   
    b2 = Vector([12.0, 13.0, 14.0]) 

    x_gauss = morph.gauss_solve(A, b)
    print("Solution by Gauss:", x_gauss)

    x_jacobi = morph.jacobi_solve(B, b2, tol=1e-6, max_iter=100)
    print("Solution by Jacobi:", x_jacobi)

    assert all(abs(x - y) < 1e-3 for x, y in zip(x_gauss, [2.0, 3.0, -1.0])), "Gauss failed"
    assert all(abs(x - y) < 1e-3 for x, y in zip(x_jacobi, [1.0, 1.0, 1.0])), "Jacobi failed"

    print("\n✅ Solver tests passed!")


def test_relativity_functions():
    import numpy as np

    # Point dans l'espace-temps (t=0, r=10, θ=π/2, φ=0)
    X_np = np.array([0.0, 10.0, np.pi / 2.0, 0.0], dtype=np.float64)

    # Création du tenseur de métrique et de son inverse via Morpheus (C++ côté)
    metric_name = "kerr"
    M = 1.0
    a = 0.8

    g_np = morph.compute_metric(X_np, metric_name, M, a)
    ginv_np = morph.inverse(g_np)

    print("\nMetric tensor g_{μν}:\n", g_np)
    print("Inverse metric g^{μν}:\n", ginv_np)

    # Calcul des symboles de Christoffel
    Gamma = morph.compute_christoffel(X_np, g_np, ginv_np, metric_name, M, a)
    print("\nChristoffel symbols Γ^λ_{μν}:\n", Gamma)

    # Calcul du tenseur de Riemann
    R = morph.compute_riemann_tensor(X_np, metric_name, M, a)
    print("\nRiemann tensor R^λ_{μνρ}:\n", R)

    # Vérification élémentaire
    assert g_np.shape == (4, 4)
    assert ginv_np.shape == (4, 4)
    assert Gamma.shape == (4, 4, 4)
    assert R.shape == (4, 4, 4, 4)

    print("\n✅ Relativity pipeline tested successfully.")


test_relativity_functions()
test_solver()

benchmark_large_matrix(8192)
