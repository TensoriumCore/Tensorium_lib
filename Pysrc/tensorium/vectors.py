from __future__ import annotations
from typing import List

from . import tensorium as _C

Vector = _C.Vector
Vectord = _C.Vectord


def add(a: Vector, b: Vector) -> Vector:
    """Add two vectors elementwise."""
    return _C.tns.add_vec(a, b)


def sub(a: Vector, b: Vector) -> Vector:
    """Subtract vector b from a."""
    return _C.tns.sub_vec(a, b)


def scale(a: Vector, factor: float) -> Vector:
    """Scale vector by a scalar."""
    return _C.tns.scl_vec(a, factor)


def dot(a: Vector, b: Vector) -> float:
    """Dot product between two vectors."""
    return _C.tns.dot_vec(a, b)


__all__ = [
    "Vector",
    "Vectord",
    "add",
    "sub",
    "scale",
    "dot",
]
