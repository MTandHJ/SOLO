

import torch, math

from torchao.prototype.low_bit_optim.subclass_8bit import OptimState8bit
from torchao.prototype.low_bit_optim.quant_utils import create_dynamic_map
from .utils import set_block_size


QMAP_SIGNED_LINEAR = torch.linspace(-1, 1, 256).tolist()
QMAP_UNSIGNED_LINEAR = torch.linspace(0, 1, 257)[1:].tolist()

QMAP_SIGNED_DE = create_dynamic_map(signed=True, max_exponent_bits=7, total_bits=8)
QMAP_UNSIGNED_DE = create_dynamic_map(signed=False, max_exponent_bits=7, total_bits=8)


class FixedLinearOptimState8bit(OptimState8bit):

    @classmethod
    def zeros(cls, shape, signed: bool = True, block_size: int = 256, device=None, **other_kwargs):
        shape = (shape,) if isinstance(shape, int) else shape
        n_elems = math.prod(shape)
        block_size = set_block_size(n_elems, block_size)

        codes = torch.zeros(shape, dtype=torch.uint8, device=device)
        scale = torch.zeros(codes.numel() // block_size, device=device)
        qmap = torch.tensor(QMAP_SIGNED_LINEAR if signed else QMAP_UNSIGNED_LINEAR, device=device)
        return cls(codes, scale, qmap, signed)


class FixedDEOptimState8bit(OptimState8bit):

    @classmethod
    def zeros(cls, shape, signed: bool = True, block_size: int = 256, device=None, **other_kwargs):
        shape = (shape,) if isinstance(shape, int) else shape
        n_elems = math.prod(shape)
        block_size = set_block_size(n_elems, block_size)

        codes = torch.zeros(shape, dtype=torch.uint8, device=device)
        scale = torch.zeros(codes.numel() // block_size, device=device)
        qmap = torch.tensor(QMAP_SIGNED_DE if signed else QMAP_UNSIGNED_DE, device=device)
        return cls(codes, scale, qmap, signed)