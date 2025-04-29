

import torch, math

from .adamw import AdamWQ


class AdaBeliefQ(AdamWQ):

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
                grad_residual = grad_f32 - exp_avg_f32
                exp_avg_sq_f32 = exp_avg_sq.float().lerp(grad_residual.square(), 1 - beta2)
                denom = exp_avg_sq_f32.sqrt().div(bias_correction2_sqrt).add(group['eps'])

                exp_avg.copy_(exp_avg_f32)
                exp_avg_sq.copy_(exp_avg_sq_f32)

                step_size = group['lr'] / bias_correction1
                p_f32 = p_f32.addcdiv(exp_avg_f32, denom, value=-step_size)

                p.copy_(p_f32)

        return loss
