from __future__ import annotations

from typing import Union

from . import tensorium as _C

Matrix = _C.Matrix
Matrixd = _C.Matrixd
MatrixLike = Union[Matrix, Matrixd]


def from_numpy(values, dtype: str = "float64") -> MatrixLike:
    """Create a Tensorium matrix from a 2D NumPy array."""
    if dtype == "float32":
        return Matrix.from_numpy(values)
    if dtype == "float64":
        return Matrixd.from_numpy(values)
    raise ValueError("dtype must be 'float32' or 'float64'")


def add(a: MatrixLike, b: MatrixLike) -> MatrixLike:
    """Add two matrices."""
    return _C.tns.add_mat(a, b)


def sub(a: MatrixLike, b: MatrixLike) -> MatrixLike:
    """Subtract matrix b from a."""
    return _C.tns.sub_mat(a, b)


def scale(a: MatrixLike, factor: float) -> MatrixLike:
    """Scale matrix."""
    return _C.tns.scl_mat(a, factor)


def lerp(a: MatrixLike, b: MatrixLike, t: float) -> MatrixLike:
    """Linear interpolation."""
    return _C.tns.lerp_mat(a, b, t)


def mul(a: MatrixLike, b: MatrixLike) -> MatrixLike:
    """Matrix multiplication."""
    return _C.tns.mul(a, b)


def transpose(a: MatrixLike) -> MatrixLike:
    """Transpose matrix."""
    return _C.tns.transpose_mat(a)


def trace(a: MatrixLike) -> MatrixLike:
    """Trace matrix (returns 1x1 matrix)."""
    return _C.tns.trace_mat(a)


def mul_vec(a: MatrixLike, x):
    """Matrix-vector multiplication."""
    return _C.tns.mul_vec(a, x)


def inverse(a: MatrixLike) -> MatrixLike:
    """Inverse matrix."""
    return _C.tns.inverse_mat(a)


def det(a: MatrixLike) -> float:
    """Determinant."""
    return _C.tns.det_mat(a)


def rank(a: MatrixLike) -> int:
    """Matrix rank."""
    return _C.tns.rank_mat(a)


def gauss_solve(a: MatrixLike, b):
    """Solve Ax=b with Gauss elimination."""
    return _C.tns.gauss_solve(a, b)


def jacobi_solve(a: MatrixLike, b, tol: float = 1e-6, max_iter: int = 1000):
    """Solve Ax=b with Jacobi iterations."""
    return _C.tns.jacobi_solve(a, b, tol, max_iter)


def row_echelon(a: MatrixLike, b=None, eps: float = 1e-12) -> None:
    """In-place row-echelon reduction of A (and optionally b)."""
    if b is None:
        _C.tns.row_echelon(a, eps)
    else:
        _C.tns.row_echelon(a, b, eps)


__all__ = [
    "Matrix",
    "Matrixd",
    "MatrixLike",
    "from_numpy",
    "add",
    "sub",
    "scale",
    "lerp",
    "mul",
    "transpose",
    "trace",
    "mul_vec",
    "inverse",
    "det",
    "rank",
    "gauss_solve",
    "jacobi_solve",
    "row_echelon",
]
