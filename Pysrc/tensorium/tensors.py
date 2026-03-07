from __future__ import annotations

from . import tensorium as _C

Tensor2d = _C.tns.Tensor2d
Tensor4d = _C.tns.Tensor4d
Metric = _C.tns.Metric


def contract(t: Tensor4d, i: int, j: int):
    """Contract a rank-4 tensor along axes (i, j)."""
    return _C.tns.contract_tensor(t, i, j)


def tensor_product(a: Tensor2d, b: Tensor2d):
    """Tensor outer product of two rank-2 tensors."""
    return _C.tns.tensor_product(a, b)


def inverse_metric(g: Tensor2d) -> Tensor2d:
    """Inverse of a rank-2 tensor."""
    return _C.tns.inv_mat_tensor(g)


def compute_christoffel(x, g: Tensor2d, g_inv: Tensor2d, metric_type: str, mass: float, spin: float):
    """Compute Christoffel symbols for the provided metric setup."""
    return _C.tns.compute_christoffel(x, g, g_inv, metric_type, mass, spin)


def compute_riemann_tensor(x, metric_type: str, mass: float, spin: float):
    """Compute the Riemann tensor from metric parameters."""
    return _C.tns.compute_riemann_tensor(x, metric_type, mass, spin)


def contract_riemann_to_ricci(riemann: Tensor4d, g_inv: Tensor2d) -> Tensor2d:
    """Contract a Riemann tensor into the Ricci tensor."""
    return _C.tns.contract_riemann_to_ricci(riemann, g_inv)


def compute_ricci_scalar(ricci: Tensor2d, g_inv: Tensor2d) -> float:
    """Compute the Ricci scalar."""
    return _C.tns.compute_ricci_scalar(ricci, g_inv)

__all__ = [
    "Tensor2d",
    "Tensor4d",
    "Metric",
    "contract",
    "tensor_product",
    "inverse_metric",
    "compute_christoffel",
    "compute_riemann_tensor",
    "contract_riemann_to_ricci",
    "compute_ricci_scalar",
]
