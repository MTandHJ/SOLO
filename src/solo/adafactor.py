

import math
from typing import Any, Dict, Tuple

import torch
from torch import Tensor
from torch.distributed._tensor import DTensor
from .adamw import Optimizer, QUANTIZERS, PACKABLE, init_adaq_generator

Eps2 = Tuple[float, float]
ParamGroup = Dict[str, Any]


class AdafactorQ(Optimizer):

    r"""
    Adafactor with Quantized states.

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

    Examples:
    ---------
    >>> model: torch.nn.Module
    >>> from solo.adafactor import AdafactorQ
    >>> optimizer = AdafactorQ(
        model.parameters(),
        lr = 0.001,
        bits = (4, 2),
        quantile = 0.1,
        block_sizes = (128, 128),
        quantizers = ('de', 'qema'),
        min_quantizable_tensor_size = 128
    )
    """

    def __init__(
        self,
        params,
        lr = None,
        eps2: Eps2 = (1e-30, 1e-3),
        clip_threshold: float = 1.0,
        decay_rate: float = -0.8,
        beta1 = None,
        weight_decay: float = 0.0,
        scale_parameter: bool = True,
        relative_step: bool = True,
        warmup_init: bool = False,
        *,
        bits: Tuple[int] = (4, 2),
        quantile: float = 0.1,
        block_sizes: Tuple[int] = (128, 128),
        quantizers: Tuple[str] = ('de', 'qema'),
        min_quantizable_tensor_size: int = 0,
    ):
        if lr is not None and lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay)
            )

        defaults = dict(
            lr=lr,
            eps2=eps2,
            clip_threshold=clip_threshold,
            decay_rate=decay_rate,
            beta1=beta1,
            weight_decay=weight_decay,
            scale_parameter=scale_parameter,
            relative_step=relative_step,
            warmup_init=warmup_init,
            quantile=quantile
        )
        super(AdafactorQ, self).__init__(params, defaults)

        self.bits = bits
        self.quantizers = quantizers
        self.block_sizes = (int(block_sizes[0]), int(block_sizes[1]))
        self.packable_sizes = (PACKABLE[bits[0]], PACKABLE[bits[1]])
        self.min_quantizable_tensor_size = min_quantizable_tensor_size

        init_adaq_generator()

        print(self)

    def __str__(self):
        return f"AdamfactorQ: [{self.quantizers[0]}-{self.quantizers[1]}]" \
                f"[{self.bits[0]}-{self.bits[1]}]" \
                f"[{self.block_sizes[0]}-{self.block_sizes[1]}]"

    def _get_lr(self, param_group: ParamGroup, param_state) -> float:
        rel_step_sz = param_group["lr"]
        if param_group["relative_step"]:
            min_step = (
                1e-6 * param_state["step"]
                if param_group["warmup_init"]
                else 1e-2
            )
            rel_step_sz = min(min_step, 1.0 / math.sqrt(param_state["step"]))
        param_scale = 1.0
        if param_group["scale_parameter"]:
            param_scale = max(param_group["eps2"][1], param_state["RMS"])
        return param_scale * rel_step_sz

    def _get_options(
        self, param_group: ParamGroup, param_shape: Tuple[int, ...]
    ) -> Tuple[bool, bool]:
        factored = len(param_shape) >= 2
        use_first_moment = param_group["beta1"] is not None
        return factored, use_first_moment

    def _rms(self, tensor: torch.Tensor) -> float:
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    def _approx_sq_grad(
        self,
        exp_avg_sq_row: torch.Tensor,
        exp_avg_sq_col: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        r_factor = (
            (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1))
            .rsqrt_()
            .unsqueeze(-1)
        )
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        torch.mul(r_factor, c_factor, out=output)

    def _init_state(self, p: Tensor, signed: bool, quantile: float):
        local_p = p.to_local() if isinstance(p, DTensor) else p

        # follow bitsandbytes, only quantize tensors >= 4096 values
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

    def step(self, closure = None):
        r"""Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adafactor does not support sparse gradients."
                    )

                state = self.state[p]
                grad_shape = grad.shape

                factored, use_first_moment = self._get_options(
                    group, grad_shape
                )
                # State Initialization
                if len(state) == 0:
                    state["step"] = 0

                    if use_first_moment:
                        # Exponential moving average of gradient values
                        state["exp_avg"] = self._init_state(
                            grad, True, group['quantile']
                        )
                    if factored:
                        state["exp_avg_sq_row"] = self._init_state(
                            torch.zeros(grad_shape[:-1], dtype=torch.float, device=grad.device),
                            False, group['quantile']
                        )
                        state["exp_avg_sq_col"] = self._init_state(
                            torch.zeros(grad_shape[:-2] + grad_shape[-1:], dtype=torch.float, device=grad.device),
                            False, group['quantile']
                        )
                    else:
                        state["exp_avg_sq"] = self._init_state(
                            grad,
                            False, group['quantile']
                        )

                    state["RMS"] = 0

                state["step"] += 1
                state["RMS"] = self._rms(p.data)
                lr = self._get_lr(group, state)

                beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])
                update = (grad**2) + group["eps2"][0]

                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]

                    exp_avg_sq_row_f32 = exp_avg_sq_row.float().lerp(
                        update.mean(dim=-1), 1. - beta2t
                    )
                    exp_avg_sq_col_f32 = exp_avg_sq_col.float().lerp(
                        update.mean(dim=-2), 1. - beta2t
                    )

                    # Approximation of exponential moving average of square
                    # of gradient
                    self._approx_sq_grad(
                        exp_avg_sq_row_f32, exp_avg_sq_col_f32, update
                    )

                    state["exp_avg_sq_row"].copy_(exp_avg_sq_row_f32)
                    state["exp_avg_sq_col"].copy_(exp_avg_sq_col_f32)

                    update.mul_(grad)
                else:
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq_f32 = exp_avg_sq.float().lerp(
                        update, 1. - beta2t
                    )
                    torch.rsqrt(exp_avg_sq_f32, out=update).mul_(grad)

                    state["exp_avg_sq"].copy_(exp_avg_sq_f32)

                update.div_(
                    max(1.0, self._rms(update) / group["clip_threshold"])
                )
                update.mul_(lr)

                if use_first_moment:
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(group["beta1"]).add_(
                        update, alpha=1 - group["beta1"]
                    )
                    update = exp_avg

                if group["weight_decay"] != 0:
                    p.data.add_(p.data, alpha=-group["weight_decay"] * lr)

                p.data.add_(-update)

        return loss