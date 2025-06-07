from __future__ import annotations

from . import tensorium as _C

Tensor2d = _C.tns.Tensor2d
Tensor4d = _C.tns.Tensor4d


def contract(t: Tensor4d, i: int, j: int):
    """Contract a rank-4 tensor along axes (i, j)."""
    return _C.tns.contract_tensor(t, i, j)

__all__ = [
    "Tensor2d",
    "Tensor4d",
    "contract",
]
