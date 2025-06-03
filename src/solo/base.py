

from typing import Tuple

import torch
from torch import Tensor
from torch.distributed._tensor import DTensor
from torch.optim import Optimizer

from .quantizer import (
    BF16,
    LinearOptimState2bit, LinearOptimState3bit, LinearOptimState4bit, LinearOptimState8bit,
    DEOptimState2bit, DEOptimState3bit, DEOptimState4bit, DEOptimState8bit,
    LinearSROptimState2bit, LinearSROptimState3bit, LinearSROptimState4bit, LinearSROptimState8bit,
    DESROptimState2bit, DESROptimState3bit, DESROptimState4bit, DESROptimState8bit,
    QemaOptimState2bit, QemaOptimState3bit, QemaOptimState4bit, QemaOptimState8bit
)
from .quantizer.utils import init_adaq_generator


QUANTIZERS = {
    2: {
        'none': None, 
        'linear': LinearOptimState2bit, 'de': DEOptimState2bit, 
        'linear-sr': LinearSROptimState2bit, 'de-sr': DESROptimState2bit,
        'qema': QemaOptimState2bit
    },
    3: {
        'none': None, 
        'linear': LinearOptimState3bit, 'de': DEOptimState3bit, 
        'linear-sr': LinearSROptimState3bit, 'de-sr': DESROptimState3bit,
        'qema': QemaOptimState3bit
    },
    4: {
        'none': None, 
        'linear': LinearOptimState4bit, 'de': DEOptimState4bit, 
        'linear-sr': LinearSROptimState4bit, 'de-sr': DESROptimState4bit,
        'qema': QemaOptimState4bit
    },
    8: {
        'none': None, 
        'linear': LinearOptimState8bit, 'de': DEOptimState8bit, 
        'linear-sr': LinearSROptimState8bit, 'de-sr': DESROptimState8bit,
        'qema': QemaOptimState8bit
    },
    16: {
        'none': None, 
        'bf16': BF16
    },
    32: {
        'none': None, 
    },
}

PACKABLE = {
    2: 4, # uint8[4x2 bits]
    3: 2, # uint8[2x3 bits]
    4: 2, # uint8[2x4 bits]
    8: 1,
    16: 1,
    32: 1
}

class LowBitOptim(Optimizer):

    r"""
    Optimizer with Quantized states.

    Parameters:
    -----------
    bits: Tuple[int, int]
        (bits for 1st state, bits for 2nd state)
    quantile: float
        quantile for 2nd state logarithmic quantization (qema)
    block_sizes: Tuple[int, int]
        (block size for 1st state, block size for 2nd state)
        Then the block-wise quantization will be performed.
        See (Dettmers T., et al. 8-bit optimizers via block-wise quantization. ICLR, 2022.)[http://arxiv.org/abs/2110.02861] for details.
    quantizers: Tuple[str, str]
        (quantizer for 1st state, quantizer for 2nd state)
        -`none`: The orginal 32-bit state
        -`bf16`: The BF16 format
        -`de`: The dynamic exponent mapping without a stochastic rounding
        -`de-sr`: The dynamic exponent mapping with a stochastic rounding
        -`linear`: The linear mapping without a stochastic rounding
        -`linear-sr`: The linear mapping with a stochastic rounding
        -`qema`: The proposed logarithmic quantization
    min_quantizable_tensor_size: int
        A tensor whose size is less than `min_quantizable_tensor_size` will be excluded from quantization.
    """

    def __init__(
        self, 
        params, 
        defaults,
        *, 
        bits: Tuple[int] = (4, 2),
        quantile: float = 0.1,
        block_sizes: Tuple[int] = (128, 128),
        quantizers: Tuple[str] = ('de', 'qema'),
        min_quantizable_tensor_size: int = 128,
    ):
        if not 0.0 <= quantile <= 1.0:
            raise ValueError("Invalid 'quantile' parameter")

        defaults['quantile'] = quantile
        super().__init__(params, defaults)

        self.bits = bits
        self.quantizers = quantizers
        self.block_sizes = (int(block_sizes[0]), int(block_sizes[1]))
        self.packable_sizes = (PACKABLE[bits[0]], PACKABLE[bits[1]])
        self.min_quantizable_tensor_size = min_quantizable_tensor_size

        init_adaq_generator()

        print(self)

    def __str__(self):
        return f"{self.__class__.__name__}: [{self.quantizers[0]}-{self.quantizers[1]}]" \
                f"[{self.bits[0]}-{self.bits[1]}]" \
                f"[{self.block_sizes[0]}-{self.block_sizes[1]}]"

    def _init_state(self, p: Tensor, signed: bool, quantile: float):
        local_p = p.to_local() if isinstance(p, DTensor) else p

        state_idx = 0 if signed else 1
        block_size = self.block_sizes[state_idx]
        packable_size = self.packable_sizes[state_idx]
        quantizer = QUANTIZERS[self.bits[state_idx]][self.quantizers[state_idx]]
        if quantizer is not None \
            and local_p.numel() >= self.min_quantizable_tensor_size \
            and local_p.numel() % packable_size == 0:
            out = quantizer.zeros(
                shape=p.shape,
                signed=signed,
                block_size=block_size,
                device=p.device,
                quantile=quantile
            )
        else:
            out = torch.zeros_like(local_p)

        # wrap subclass in DTensor as needed
        # NOTE: local tensor may have different shapes across ranks.
        # this happens when the 1st dim is not divisible by WORLD_SIZE.
        # thus, we must supply shape (and stride) to DTensor.from_local()
        if isinstance(p, DTensor):
            out = DTensor.from_local(
                local_tensor=out,
                device_mesh=p.device_mesh,
                placements=p.placements,
                run_check=False,
                shape=p.shape,
                stride=p.stride(),
            )

        return out