

import math

import torch
from torch import Tensor
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_4,
    TORCH_VERSION_AT_LEAST_2_5,
    TorchAOBaseTensor,
)

from torchao.prototype.low_bit_optim.quant_utils import (
    dequant_with_qmap,
    scale_tensor,
)
from .utils import set_block_size, quantize_3bit_with_qmap

aten = torch.ops.aten
c10d_functional = torch.ops.c10d_functional
_c10d_functional = torch.ops._c10d_functional

QMAP_SIGNED_LINEAR = torch.linspace(-1, 1, 8).tolist()
QMAP_SIGNED_LINEAR = QMAP_SIGNED_LINEAR + [QMAP_SIGNED_LINEAR[-1]] * 8
QMAP_UNSIGNED_LINEAR = torch.linspace(0, 1, 9)[1:].tolist()
QMAP_UNSIGNED_LINEAR = QMAP_UNSIGNED_LINEAR + [QMAP_UNSIGNED_LINEAR[-1]] * 8


class LinearSROptimState3bit(TorchAOBaseTensor):
    tensor_attrs = ["codes", "scale", "qmap"]

    @staticmethod
    def __new__(cls, codes: Tensor, scale: Tensor, qmap: Tensor, signed: bool, shape):
        return Tensor._make_wrapper_subclass(cls, shape, device=codes.device)

    def __init__(self, codes: Tensor, scale: Tensor, qmap: Tensor, signed: bool, shape):
        """Create quantized 4-bit optimizer state as proposed in https://arxiv.org/abs/2309.01507

        Args
            codes: quantized and packed 4-bit data stored as uint8.
            scale: scale data for block-wise quantization.
            qmap: lookup table that maps between quantized value (code) and float value.
            signed: whether the tensor is signed or unsigned.
            shape: shape of original float tensor.

        NOTE: To get block-wise scale, the original float tensor is first reshape to (-1, block_size).
        Thus, the last dimension of the original float tensor is not necessarily divisible by block size.
        Given `codes` and `scale`, `block_size` is calculated as `codes.numel() * 2 // scale.numel()`.
        The extra `* 2` is because `codes` is 4-bit data packed in 8-bit storage.
        """
        assert codes.dtype is torch.uint8
        assert codes.ndim == 1  # flattened buffer
        assert scale.ndim == 1
        self.codes = codes
        self.scale = scale
        self.qmap = qmap
        self.signed = signed
        self._shape = shape
        self.block_size = codes.numel() * 2 // scale.numel()

    def __tensor_flatten__(self):
        return self.tensor_attrs, [self.signed, self._shape]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size=None, outer_stride=None
    ):
        return cls(
            *[tensor_data_dict[name] for name in cls.tensor_attrs], *tensor_attributes
        )

    def dequantize(self, output_dtype=None):
        codes = torch.stack([self.codes >> 4, self.codes & 0b1111], dim=-1)  # unpack
        float_data = dequant_with_qmap(codes, self.qmap, self.scale)
        if output_dtype is not None:
            float_data = float_data.to(output_dtype)
        return float_data.view(self._shape)

    @classmethod
    def zeros(cls, shape, signed: bool = True, block_size: int = 0, device=None, **other_kwargs):
        shape = (shape,) if isinstance(shape, int) else shape
        n_elems = math.prod(shape)
        block_size = set_block_size(n_elems, block_size)

        codes = torch.zeros(n_elems // 2, dtype=torch.uint8, device=device)
        scale = torch.zeros(n_elems // block_size, device=device)
        qmap = torch.tensor(QMAP_SIGNED_LINEAR if signed else QMAP_UNSIGNED_LINEAR, device=device)
        return cls(codes, scale, qmap, signed, shape)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(signed={self.signed}, block_size={self.block_size}, "
            f"shape={tuple(self.shape)}, device={self.device}, requires_grad={self.requires_grad})"
        )


# in pre-2.4, calling .to(device, dtype) will not dispatch aten._to_copy.default when
# dtype is the same but device is different. thus, we must override .to() method instead.
if not TORCH_VERSION_AT_LEAST_2_4:

    def _to(self, *args, **kwargs):
        # ignore other args/kwargs
        device = kwargs.pop("device", None)
        return LinearSROptimState3bit(
            self.codes.to(device),
            self.scale.to(device),
            self.qmap.to(device),
            self.signed,
            self.shape,
        )

    LinearSROptimState3bit.to = _to
    del _to  # make sure to not re-use


@LinearSROptimState3bit.implements(aten.copy_.default)
def _(func, types, args, kwargs):
    dst = args[0]
    src = args[1]

    if isinstance(dst, LinearSROptimState3bit) and isinstance(src, LinearSROptimState3bit):
        assert (
            dst.signed == src.signed
            and dst.block_size == src.block_size
            and dst._shape == src._shape
        )
        dst.codes.copy_(src.codes)
        dst.scale.copy_(src.scale)
        # qmap should be the same, don't need to copy

    elif isinstance(dst, LinearSROptimState3bit):
        scaled_src, scale = scale_tensor(src.view(-1), dst.block_size)
        codes = quantize_3bit_with_qmap(scaled_src, dst.qmap, stochastic_rounding=True)
        dst.codes.copy_((codes[::2] << 4) | codes[1::2])  # packing
        dst.scale.copy_(scale)

    else:
        dst.copy_(src.dequantize())

    return dst


@LinearSROptimState3bit.implements(aten._to_copy.default)
def _(func, types, args, kwargs):
    # ignore dtype
    device = kwargs.get("device", None)
    out = LinearSROptimState3bit(
        args[0].codes.to(device=device),
        args[0].scale.to(device=device),
        args[0].qmap.to(device=device),
        args[0].signed,
        args[0].shape,
    )
    return return_and_correct_aliasing(func, args, kwargs, out)


@LinearSROptimState3bit.implements(aten.lerp.Scalar)
def _(func, types, args, kwargs):
    args = [x.dequantize() if isinstance(x, LinearSROptimState3bit) else x for x in args]
    return func(*args, **kwargs)


# this is needed for DTensor.from_local() and for flattening tensor
@LinearSROptimState3bit.implements(aten.view.default)
def _(func, types, args, kwargs):
    x, shape = args

    if tuple(x.shape) == tuple(shape):
        return LinearSROptimState3bit(x.codes, x.scale, x.qmap, x.signed, x._shape)

    if len(shape) == 1 and shape[0] == -1:
        return LinearSROptimState3bit(x.codes, x.scale, x.qmap, x.signed, (x.numel(),))

    raise ValueError(
        f"{x.__class__.__name__} only supports .view() with same shape or shape=[-1]"
    )


@LinearSROptimState3bit.implements(
    [
        # required by DTensor.full_tensor()
        c10d_functional.all_gather_into_tensor.default,
        _c10d_functional.all_gather_into_tensor.default,
        c10d_functional.wait_tensor.default,
        _c10d_functional.wait_tensor.default,
        # required by torch.distributed.checkpoint.save
        aten.detach.default,
    ]
)
def _(func, types, args, kwargs):
    x = args[0]
    if not isinstance(x, LinearSROptimState3bit):
        raise ValueError(f"expecting a LinearSROptimState3bit but found {type(x)}")

    codes = func(x.codes, *args[1:], **kwargs)
    scale = func(x.scale, *args[1:], **kwargs)

    # adjust the first dim
    shape = (x._shape[0] * codes.numel() // x.codes.numel(),) + x._shape[1:]

    # assume tensors from all ranks have the same signedness
    return LinearSROptimState3bit(codes, scale, x.qmap.clone(), x.signed, shape)


# required by torch.distributed.checkpoint.save
# note that we don't actually implement pin memory for this tensor subclass
# (pin_memory argument is ignored in aten._to_copy)
@LinearSROptimState3bit.implements(aten.is_pinned.default)
def _(func, types, args, kwargs):
    return (
        args[0].codes.is_pinned()
        and args[0].scale.is_pinned()
        and args[0].qmap.is_pinned()
    )


# required by torch.distributed.checkpoint.load when world size changes i.e. re-sharding
@LinearSROptimState3bit.implements(aten.slice.Tensor)
def _(func, types, args, kwargs):
    x, dim, start, end = args[:4]
    step = args[4] if len(args) > 4 else 1

    # input validation
    if dim != 0:
        raise ValueError("Only support aten.slice along the first dim")
    if step != 1:
        raise ValueError("Only support aten.slice with step=1")

    block_size = x.block_size
    stride = math.prod(x.shape[1:])

    # for 1 increment in x along the first dim,
    # (flattened) scale will increment by stride / block_size
    if (start * stride) % block_size != 0 or (end * stride) % block_size != 0:
        raise ValueError(
            f"Invalid start or end for shape={x.shape} and block_size={block_size}. "
            f"Make sure start and end align with block boundary. "
            f"Received start={start}, end={end}."
        )

    # note that for 4-bit, we store .codes as flattened buffer
    # divide by 2 since we store 2x 4-bit in 1x uint8
    codes = x.codes[start * stride // 2 : end * stride // 2]
    scale = x.scale[start * stride // block_size : end * stride // block_size]

    # adjust the first dim
    shape = (x.shape[0] * codes.numel() // x.codes.numel(),) + x.shape[1:]

    return LinearSROptimState3bit(codes, scale, x.qmap.clone(), x.signed, shape)


if TORCH_VERSION_AT_LEAST_2_5:
    from torch.serialization import add_safe_globals

    add_safe_globals([LinearSROptimState3bit])
