

import torch


class BF16:

    @classmethod
    def zeros(cls, shape, signed: bool = True, block_size: int = 0, device=None, **other_kwargs):
        return torch.zeros(shape, dtype=torch.bfloat16, device=device)