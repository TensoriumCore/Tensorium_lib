from __future__ import annotations
from typing import Sequence, Union

from . import tensorium as _C

Vector = _C.Vector
Vectord = _C.Vectord
VectorLike = Union[Vector, Vectord]


def from_numpy(values, dtype: str = "float64") -> VectorLike:
    """Create a Tensorium vector from a 1D NumPy array."""
    if dtype == "float32":
        return Vector.from_numpy(values)
    if dtype == "float64":
        return Vectord.from_numpy(values)
    raise ValueError("dtype must be 'float32' or 'float64'")


def add(a: VectorLike, b: VectorLike) -> VectorLike:
    """Add two vectors elementwise."""
    return _C.tns.add_vec(a, b)


def sub(a: VectorLike, b: VectorLike) -> VectorLike:
    """Subtract vector b from a."""
    return _C.tns.sub_vec(a, b)


def scale(a: VectorLike, factor: float) -> VectorLike:
    """Scale vector by a scalar."""
    return _C.tns.scl_vec(a, factor)


def dot(a: VectorLike, b: VectorLike) -> float:
    """Dot product between two vectors."""
    return _C.tns.dot_vec(a, b)


def norm1(a: VectorLike) -> float:
    """L1 norm."""
    return _C.tns.norm_1(a)


def norm2(a: VectorLike) -> float:
    """L2 norm."""
    return _C.tns.norm_2(a)


def norm_inf(a: VectorLike) -> float:
    """Infinity norm."""
    return _C.tns.norm_inf(a)


def cosine(a: VectorLike, b: VectorLike) -> float:
    """Cosine similarity."""
    return _C.tns.cosine(a, b)


def lerp(a: VectorLike, b: VectorLike, t: float) -> VectorLike:
    """Linear interpolation."""
    return _C.tns.lerp(a, b, t)


def linear_comb(vectors: Sequence[VectorLike], coeffs: Sequence[float]) -> VectorLike:
    """Linear combination sum(c_i * v_i)."""
    return _C.tns.linear_comb(list(vectors), list(coeffs))


def cross(a: VectorLike, b: VectorLike) -> VectorLike:
    """3D cross product."""
    return _C.tns.cross(a, b)


__all__ = [
    "Vector",
    "Vectord",
    "VectorLike",
    "from_numpy",
    "add",
    "sub",
    "scale",
    "dot",
    "norm1",
    "norm2",
    "norm_inf",
    "cosine",
    "lerp",
    "linear_comb",
    "cross",
]
