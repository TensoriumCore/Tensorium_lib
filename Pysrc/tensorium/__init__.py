"""Python interface to the Tensorium C++ bindings."""
from __future__ import annotations

from . import tensorium as _C

from . import vectors
from . import matrices
from . import tensors

# Re-export commonly used classes
Vector = vectors.Vector
Vectord = vectors.Vectord
Matrix = matrices.Matrix
Matrixd = matrices.Matrixd
Tensor2d = tensors.Tensor2d
Tensor4d = tensors.Tensor4d

__all__ = [
    "Vector",
    "Vectord",
    "Matrix",
    "Matrixd",
    "Tensor2d",
    "Tensor4d",
    "vectors",
    "matrices",
    "tensors",
]
