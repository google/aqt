# AQT : Accurate Quantized Training

AQT is a software library designed for easy tensor opeartion quantization in JAX. AQT simultaneously provides:

  * excellent quantized model quality with no hand-tuning,
  * excellent training performance in production using contemporary ML accelerators,
  * simple and flexible APIs suitable for both production and research.

AQT is designed for both quantization researchers and production workloads. It has the following features:

  * What you train is what you serve. AQT quantized models are bit-exact the same during training and serving. This side-steps the conventional issue of quantization-induced training-serving bias that typically happens for Post Training Quantization (PTQ).
  * JAX universal and easy to use. AQT leverages quantization injection to quantize all JAX tensor ops. The injection method has been adopted by [Flax](https://github.com/google/flax), [Pax](https://github.com/google/paxml), and other frameworks at Google.

**Note:  Users are recommended to use `aqt.jax.v2`. Other jax versions are obsolete.**

## Usage

Tensor contraction operations in JAX-based neural network libraries, i.e., any form of (high-order) matrix multiplications, including but not limited to `jax.numpy.einsum` and `flax.linen.DenseGeneral`, call `lax.dot_general` as its core computation. Quantizing a neural network in JAX simply requires substituting `lax.dot_general` with a quantized variant and keeping other parts as-is, which we call "quantization injection". JAX-based NN libraries, such as [Flax](https://github.com/google/flax) and [Pax](https://github.com/google/paxml), provide an API for this substitution when creating layers.

In this section, we show how AQT produces a quantized `dot_general` and inject it into a neural network defined in JAX. The toy example below can be found in [examples.ipynb](./aqt/jax/v2/examples/examples.ipynb).

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

## Flexible Quantization Configs

The example in [usage](#usage) uses the default configuration that quantizes both forward and backward passes to 8-bit, but AQT provides a much more flexible configuration system. The `DotGeneral` class can configure forward and backward tensor contraction operations separately.
```python
@dataclasses.dataclass
class DotGeneral:
  """Configuration of quantization of dot_general and its gradients."""
  fwd: DotGeneralRaw
  dlhs: DotGeneralRaw
  drhs: DotGeneralRaw
```

In each `DotGeneral.DotGeneralRaw`, we can configure quantization of each input tensor of those ops separately and the hardware dtype to use (eg. jnp.bfloat16, jnp.float16, jnp.float8_e4m3fn, jnp.float8_e5m2, jnp.int8, jnp.int4).
```python
@dataclasses.dataclass
class DotGeneralRaw:
  """Configuration of quantization of one dot_general without gradient."""
  lhs: Tensor  # left hand side
  rhs: Tensor  # right hand side
  dg_in_dtype: Optional[DType]
  dg_accumulator_dtype: Optional[DType]
  local_aqt: Optional[LocalAqt]  # sharded quantization
```

Inside config.Tensor we can configure the numerics used for each tensor, which includes number of bits, calibration algorithm, stochastic rounding, and many other quantization parameters.
```python
@dataclasses.dataclass
class Tensor:
  """Configuration of quantization of one tensor or one side of tensor op."""
  numerics: Numerics
  calib_shared_axes: Optional[list[int]]
  scale_stop_grad: bool
  calibration: calibration.Calibration  # calibration algorithm
  po2_scale: bool  # round calibration to power of 2
  use_fake_quant: bool
  use_fwd_quant: Optional[bool]  # use quantized fwd in the bwd pass
```

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
