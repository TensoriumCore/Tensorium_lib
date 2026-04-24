"""Python interface to the Tensorium C++ bindings."""
from __future__ import annotations

from . import tensorium as _C

from . import vectors
from . import matrices
from . import tensors
from . import bssn
from . import z4c

# Re-export commonly used classes
Vector = vectors.Vector
Vectord = vectors.Vectord
Matrix = matrices.Matrix
Matrixd = matrices.Matrixd
Tensor2d = tensors.Tensor2d
Tensor4d = tensors.Tensor4d
Z4cGrid = z4c.Z4cGrid
Z4cRKStepper = z4c.Z4cRKStepper
BSSNGrid = Z4cGrid
BSSNRKStepper = Z4cRKStepper

__all__ = [
    "Vector",
    "Vectord",
    "Matrix",
    "Matrixd",
    "Tensor2d",
    "Tensor4d",
    "Z4cGrid",
    "Z4cRKStepper",
    "BSSNGrid",
    "BSSNRKStepper",
    "vectors",
    "matrices",
    "tensors",
    "z4c",
    "bssn",
]
