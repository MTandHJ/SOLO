

import torch

from torch import Tensor


def scale_tensor(input: Tensor, block_size: int, quantile: float, eps: float = 1.e-12):
    shape = input.shape

    input = input.view(-1, block_size)
    scale = input.amax(-1).clamp_min(eps)
    alpha = input.quantile(quantile, -1) / scale
    input = input / scale.view(-1, 1)
    return input.view(shape), scale, alpha

def quantize_with_alpha(input: Tensor, alpha: Tensor, bits: int = 4):
    shape = input.shape

    logalpha = alpha.log2().view(-1, 1)
    input = input.view(logalpha.size(0), -1)
    codes = input.log2().div(logalpha).add(
        torch.rand(input.size(), generator=adaq_generator, device=input.device) - 0.5
    ).round().clip(0, 2 ** bits - 1)
    return codes.view(shape).to(torch.uint8)

def dequant_with_alpha(codes: Tensor, alpha: Tensor, scale: Tensor):
    out = (alpha.view(-1, 1) ** codes.view(alpha.size(0), -1)).mul(scale.view(-1, 1))
    return out.view(codes.shape)

def quantize_2bit_with_qmap(input: Tensor, qmap: Tensor, stochastic_rounding: bool = False):
    # GPU-friendly binary search
    # https://blog.demofox.org/2017/06/20/simd-gpu-friendly-branchless-binary-search/
    codes = torch.where(input >= qmap[2], 2, 0)
    codes += torch.where(input >= qmap[codes + 1], 1, 0)

    # rounding
    codes = codes.clip(max=3)
    codes_up = (codes + 1).clip(max=3)
    val_down = qmap[codes]
    val_up = qmap[codes_up]
    residual = input - val_down

    p = val_up - val_down
    if stochastic_rounding:
        p = p.mul(torch.rand(residual.size(), generator=adaq_generator, device=input.device))
    else:
        p = p * 0.5
    codes = torch.where(residual >= p, codes_up, codes)

    return codes.to(torch.uint8)

def quantize_3bit_with_qmap(input: Tensor, qmap: Tensor, stochastic_rounding: bool = False):
    # GPU-friendly binary search
    # https://blog.demofox.org/2017/06/20/simd-gpu-friendly-branchless-binary-search/
    codes = torch.where(input >= qmap[8], 8, 0)
    codes += torch.where(input >= qmap[codes + 4], 4, 0)
    codes += torch.where(input >= qmap[codes + 2], 2, 0)
    codes += torch.where(input >= qmap[codes + 1], 1, 0)

    # rounding
    codes = codes.clip(max=7)
    codes_up = (codes + 1).clip(max=7)
    val_down = qmap[codes]
    val_up = qmap[codes_up]
    residual = input - val_down

    p = val_up - val_down
    if stochastic_rounding:
        p = p.mul(torch.rand(residual.size(), generator=adaq_generator, device=input.device))
    else:
        p = p * 0.5
    codes = torch.where(residual >= p, codes_up, codes)

    return codes.to(torch.uint8)

def quantize_4bit_with_qmap(input: Tensor, qmap: Tensor, stochastic_rounding: bool = False):
    # GPU-friendly binary search
    # https://blog.demofox.org/2017/06/20/simd-gpu-friendly-branchless-binary-search/
    codes = torch.where(input >= qmap[8], 8, 0)
    codes += torch.where(input >= qmap[codes + 4], 4, 0)
    codes += torch.where(input >= qmap[codes + 2], 2, 0)
    codes += torch.where(input >= qmap[codes + 1], 1, 0)

    # rounding
    codes = codes.clip(max=15)
    codes_up = (codes + 1).clip(max=15)
    val_down = qmap[codes]
    val_up = qmap[codes_up]
    residual = input - val_down

    p = val_up - val_down
    if stochastic_rounding:
        p = p.mul(torch.rand(residual.size(), generator=adaq_generator, device=input.device))
    else:
        p = p * 0.5
    codes = torch.where(residual >= p, codes_up, codes)

    return codes.to(torch.uint8)

def quantize_8bit_with_qmap(input: Tensor, qmap: Tensor, stochastic_rounding: bool = False):
    # GPU-friendly binary search
    # https://blog.demofox.org/2017/06/20/simd-gpu-friendly-branchless-binary-search/
    codes = torch.where(input >= qmap[128], 128, 0)
    codes += torch.where(input >= qmap[codes + 64], 64, 0)
    codes += torch.where(input >= qmap[codes + 32], 32, 0)
    codes += torch.where(input >= qmap[codes + 16], 16, 0)
    codes += torch.where(input >= qmap[codes + 8], 8, 0)
    codes += torch.where(input >= qmap[codes + 4], 4, 0)
    codes += torch.where(input >= qmap[codes + 2], 2, 0)
    codes += torch.where(input >= qmap[codes + 1], 1, 0)

    # rounding
    codes_up = (codes + 1).clip(max=255)
    val_down = qmap[codes]
    val_up = qmap[codes_up]
    residual = input - val_down

    p = val_up - val_down
    if stochastic_rounding:
        p = p.mul(torch.rand(residual.size(), generator=adaq_generator, device=input.device))
    else:
        p = p * 0.5
    codes = torch.where(residual >= p, codes_up, codes)

    return codes.to(torch.uint8)

def init_adaq_generator():
    global adaq_generator

    if torch.distributed.is_initialized():
        seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        torch.distributed.broadcast(seed, src=0)
    else:
        seed = torch.randint(1 << 31, size=[])
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    adaq_generator = torch.Generator(device=rank)
    adaq_generator.manual_seed(seed.item())

def set_block_size(n_elems: int, block_size: int) -> int:
    block_size = n_elems if block_size <= 0 else block_size
    block_size = n_elems if n_elems <= block_size else block_size
    for b in range(block_size, 0, -1):
        if n_elems % b == 0:
            return b
    return 1