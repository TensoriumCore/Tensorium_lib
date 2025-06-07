from __future__ import annotations

from . import tensorium as _C

Matrix = _C.Matrix


def add(a: Matrix, b: Matrix) -> Matrix:
    """Add two matrices."""
    return _C.tns.add_mat(a, b)


def sub(a: Matrix, b: Matrix) -> Matrix:
    """Subtract matrix b from a."""
    return _C.tns.sub_mat(a, b)


def mul(a: Matrix, b: Matrix) -> Matrix:
    """Matrix multiplication."""
    return _C.tns.mul(a, b)

__all__ = ["Matrix", "add", "sub", "mul"]
