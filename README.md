

<div align="center">
  <img src="assets/logo.svg" alt="Logo">
</div>

<h5 align="center">
    <p>
        <a href="">Paper</a> |
        <a href="https://pytorch.org/">PyTorch >= 2.3</a> |
        <a href="https://github.com/pytorch/ao/tree/main">torchao >= 0.7.0</a>
    </p>
</h4>


## Installation

```
pip install solow
```

or

```
pip install git+https://github.com/MTandHJ/SOLO.git
```

## Usage


```python
from solo.adamw import AdamWQ

optimizer = AdamWQ(
    model.parameters(),
    lr = 0.001,
    weight_decay = 0.,
    betas = (0.8, 0.999),
    bits = (4, 2), # (4 bits for signed, 2 bits for unsigned)
    quantile = 0.1,
    block_sizes = (128, 128),
    quantizers = ('de', 'qema'),
    # A tensor whose size is less than `min_quantizable_tensor_size`
    # will be excluded from quantization.
    # For rigorous probing, this value is set to 0 in paper.
    # Assigning a larger value (such as the default of 4096 in torchao) 
    # may yield more stable results.
    min_quantizable_tensor_size = 0
)

```

- `quantizers`:
    - `none`: The orginal 32-bit state.
    - `bf16`: The BF16 format.
    - `de`: The dynamic exponent mapping without a stochastic rounding.
    - `de-sr`: The dynamic exponent mapping with a stochastic rounding.
    - `linear`: The linear mapping without a stochastic rounding.
    - `linear-sr`: The linear mapping with a stochastic rounding.
    - `qema`: The proposed logarithmic quantization.


## Reference Code

- [pytorch-optimizer](https://github.com/jettify/pytorch-optimizer/tree/master): We implemented the low-bit Adafactor and AdaBelief optimiers based on this code.

