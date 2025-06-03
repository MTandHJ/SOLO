
from typing import Tuple

import torch, math

from .base import LowBitOptim


class AdamWQ(LowBitOptim):

    r"""
    AdamW with Quantized states.

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
    >>> from solo.adamw import AdamWQ
    >>> optimizer = AdamWQ(
        model.parameters(),
        lr = 0.001,
        weight_decay = 0.,
        betas = (0.8, 0.999),
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
        lr: float = 0.001, 
        weight_decay: float = 0., 
        betas: Tuple[float] = (0.9, 0.999),
        eps: float = 1e-8, 
        *, 
        bits: Tuple[int] = (4, 2),
        quantile: float = 0.1,
        block_sizes: Tuple[int] = (128, 128),
        quantizers: Tuple[str] = ('de', 'qema'),
        min_quantizable_tensor_size: int = 128,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )

        super().__init__(
            params, defaults,
            bits=bits,
            quantile=quantile,
            block_sizes=block_sizes,
            quantizers=quantizers,
            min_quantizable_tensor_size=min_quantizable_tensor_size
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Sparse gradient is not supported")

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0)
                    state["exp_avg"] = self._init_state(p, True, group['quantile'])
                    state["exp_avg_sq"] = self._init_state(p, False, group['quantile'])

                state["step"] += 1
                step = state["step"].item()

                p_f32 = p.float()
                grad_f32 = grad.float()
                
                # weight decay
                p_f32 = p_f32.mul(1 - group['lr'] * group['weight_decay'])

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                beta1, beta2 = group['betas']
                bias_correction1 = 1 - beta1 ** step
                bias_correction2_sqrt = math.sqrt(1 - beta2 ** step)

                # keep high precision copy for param update
                exp_avg_f32 = exp_avg.float().lerp(grad_f32, 1 - beta1)
                exp_avg_sq_f32 = exp_avg_sq.float().lerp(grad_f32.square(), 1 - beta2)
                denom = exp_avg_sq_f32.sqrt().div(bias_correction2_sqrt).add(group['eps'])

                exp_avg.copy_(exp_avg_f32)
                exp_avg_sq.copy_(exp_avg_sq_f32)

                step_size = group['lr'] / bias_correction1
                p_f32 = p_f32.addcdiv(exp_avg_f32, denom, value=-step_size)

                p.copy_(p_f32)

        return loss