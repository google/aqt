# AQT : Accurate Quantized Training

AQT is a quantization library designed to allow utilization of
low-bit and high-performance numerics of contemporary ML hardware accelerators.
AQT supports both research and production[^research-vs-prod], but focuses on the latter.

[^research-vs-prod]: The support for research is exemplified by having a state of the art quantization quality on standard models such as ResNet and Transformer. The production aspect is defined as high performance and robust out-of-the-box working results with good defaults.

## Usage

Tensor contraction operations in JAX-based neural network libraries, i.e., any form of (high-order) matrix multiplications, including but not limited to `jax.numpy.einsum` and `flax.linen.DenseGeneral`, call `lax.dot_general` as its core computation. Quantizing a neural network in JAX simply requires substituting `lax.dot_general` with a quantized variant and keeping other parts as-is, which we call "quantization injection". JAX-based NN libraries, such as [Flax](https://github.com/google/flax) and [Pax](https://github.com/google/paxml), provide an API for this substitution when creating layers.

In this section, we show how AQT produces a quantized `dot_general` and inject it into a neural network defined in JAX. The toy example below can be found in [examples.ipynb](./examples.ipynb).

First, install the AQT package named as `aqtp` in PyPI and import necessary files.
```python
# install the AQT library
!pip install aqtp
# necessary imports
import aqt.jax.v2.aqt_dot_general as aqt
import aqt.jax.v2.config as aqt_config
```

Next, specify an AQT configuration that quantizes both forward and backward passes to int8.
```python
# create a config that quantizes both forward and backward passes to int8
int8_config = aqt_config.fully_quantized(fwd_bits=8, bwd_bits=8)
```

A [sample neural network](https://github.com/google/flax/blob/abab11fff54e229cb2691ebf71a7515abbd0a547/examples/lm1b/models.py#L170) defined in Flax looks like the following (as a toy example we use a simple MLP, but it can be any model):
```python
class MlpBlock(nn.Module):
  config: aqt_config.DotGeneral | None

  @nn.compact
  def __call__(self, inputs):
    dot_general = aqt.make_dot_general(self.config)
    x = nn.Dense(dot_general=dot_general, features=inputs.shape[-1] * 4)(inputs)
    x = nn.relu(x)
    x = nn.Dense(dot_general=dot_general, features=inputs.shape[-1])(x)
    return x
```

AQT can quantize the model by simply replacing the `dot_general` in `nn.Dense` with a quantized dot_general created by the aqt configuration.

Now let's test it.
```python
import jax
import jax.numpy as jnp
import numpy as np

# Generate some random matrices as inputs
def gen_matrix(rows, columns, seed=0):
  np.random.seed(seed)
  return np.random.normal(size=(rows, columns)).reshape((rows, columns))

inputs = gen_matrix(3, 4)

# test function that initializes the model and compute the forward pass
def init_and_eval(name, mlp_block, init_seed=0):
  model = mlp_block.init(jax.random.PRNGKey(init_seed), inputs)
  out = mlp_block.apply(model, inputs)
  print(f"{name}:\n", out)

# run and print results
mlp_fp16 = MlpBlock(config=None)
mlp_int8 = MlpBlock(config=int8_config)
init_and_eval('mlp_fp16', mlp_fp16)
init_and_eval('mlp_int8', mlp_int8)
```

Results will be the following:
```
mlp_fp16:
 [[ 0.720744    1.5375545  -2.6456933  -1.7605033 ]
 [-0.01541612  0.09728499 -1.5742414  -0.3737522 ]
 [ 0.4071759   1.1941448  -0.6982092  -0.48336366]]
mlp_int8:
 [[ 0.7030779   1.5099456  -2.6334763  -1.7550919 ]
 [-0.00901393  0.08774488 -1.5644912  -0.3728472 ]
 [ 0.40121436  1.189411   -0.6939187  -0.48000643]]
```

We can see that the quantized MLP produces similar outputs as the unquantized one.


## Citing AQT
Please use a following bibtex entry:

```
@software{aqt2022github,
  author = {Lew, Lukasz and Feinberg, Vlad and Agrawal, Shivani and Lee, Jihwan and Malmaud, Jonathan and Wang, Lisa and  Dormiani, Pouya and Pope, Reiner },
  title = {AQT: Accurate Quantized Training)},
  url = {http://github.com/google/aqt},
  year = {2022},
}
```
