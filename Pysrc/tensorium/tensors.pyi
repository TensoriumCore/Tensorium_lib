from .tensorium import tns

Tensor2d = tns.Tensor2d
Tensor4d = tns.Tensor4d

def contract(t: Tensor4d, i: int, j: int): ...

__all__: list[str]
