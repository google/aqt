# AQT : Accurate Quantized Training

AQT is a software library designed for easy tensor operation quantization in JAX.

## Features

  * Excellent quantized int8 model quality with no hand-tuning at all.
  * Excellent training speedup in production using contemporary ML accelerators.
  * Simple and flexible APIs suitable for both production and research.
  * AQT is a tensor op (matmul, einsum, conv, ...) quantization library.
    Does not make assumptions about the computations to be Neural Net.
  * AQT can be injected into any Jax computation.
    Well tested in:
    - [Flax](https://github.com/google/flax),
    - [Pax](https://github.com/google/paxml),
    - [MaxText](https://github.com/google/maxtext)
    - and other frameworks at Google.
  * AQT is designed for production workloads.
    It is well exemplified by these blog entries:
    - [Cloud ML AQT introduction](https://cloud.google.com/blog/products/compute/accurate-quantized-training-aqt-for-tpu-v5e/)
    - [50000+ TPU Cloud Ml training with AQT](https://cloud.google.com/blog/products/compute/the-worlds-largest-distributed-llm-training-job-on-tpu-v5e)
  * AQT is designed for research.
    It is well exemplified by these papers:
    - [Pareto-Optimal Quantized ResNet Is Mostly 4-bit](https://arxiv.org/abs/2105.03536)
    - [PokeBNN: A Binary Pursuit of Lightweight Accuracy](https://arxiv.org/abs/2112.00133)
    - [4-bit Conformer with Native Quantization Aware Training for Speech Recognition](https://arxiv.org/abs/2203.15952)
    - [Binarized Neural Machine Translation](https://arxiv.org/abs/2302.04907)
  * WYTIWYS (What you train is what you serve).
    AQT quantized models are bit-exact the same during training and serving.
    This side-steps the conventional issue of quantization-induced training-serving bias
    that typically happens for Post Training Quantization (PTQ).


Let us know if you have any problem with aqt applications by filing an issue on Github.


## Usage

Tensor contraction operations in JAX-based neural network libraries, i.e., any form of (high-order) matrix multiplications, including but not limited to `jax.numpy.einsum` and `flax.linen.DenseGeneral`, call [lax.dot_general](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.dot_general.html) as its core computation. Quantizing a neural network in JAX simply requires substituting `lax.dot_general` with a quantized variant and keeping other parts as-is, which we call "quantization injection". JAX-based NN libraries, such as [Flax](https://github.com/google/flax) and [Pax](https://github.com/google/paxml), provide an API for this substitution when creating layers.

In this section, we show how AQT produces a quantized `dot_general` and inject it into a neural network defined in JAX. The toy example below can be found in the [example colab](https://github.com/google/aqt/blob/main/aqt/jax/v2/examples/examples.ipynb).

First, install the AQT package named as `aqtp` in PyPI.

```python
# install the AQT library
!pip install aqtp
```

Next, import `aqt.jax.v2`.
Other [AQT versions](#aqt-versions) are obsolete.

```python
# necessary imports
import aqt.jax.v2.flax.aqt_flax as aqt
import aqt.jax.v2.config as aqt_config
import flax.linen as nn
```

A [sample neural network](https://github.com/google/flax/blob/abab11fff54e229cb2691ebf71a7515abbd0a547/examples/lm1b/models.py#L170) defined in Flax looks like the following (as a toy example we use a simple MLP, but it can be any model):

```python
class MlpBlock(nn.Module):
  config: aqt_config.DotGeneral | None

  @nn.compact
  def __call__(self, inputs):
    dot_general = aqt.AqtDotGeneral(self.config)
    x = nn.Dense(dot_general=dot_general, features=inputs.shape[-1] * 4)(inputs)
    x = nn.relu(x)
    x = nn.Dense(dot_general=dot_general, features=inputs.shape[-1])(x)
    return x
```

AQT can quantize the model by simply replacing the `dot_general` in `nn.Dense` with a quantized dot_general created by the aqt configuration. The example specifies an AQT configuration that quantizes both forward and backward passes to int8. Now let's test it.

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
def init_and_eval(name, mlp_block, init_seed=0, eval_seed=0):
  model = mlp_block.init(jax.random.PRNGKey(init_seed), inputs)
  out = mlp_block.apply(model, inputs, rngs={'params': jax.random.key(eval_seed)})
  print(f"{name}:\n", out)

# create a config that quantizes both forward and backward passes to int8
int8_config = aqt_config.fully_quantized(fwd_bits=8, bwd_bits=8)

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

## How to make and use a simple AQT quantizer

This example demonstrates how to make a simple linear AQT quantizer.

```python
from aqt.jax.v2 import aqt_quantizer
from aqt.jax.v2 import calibration
from aqt.jax.v2 import utils as aqt_utils
from aqt.jax.v2.numerics import int_numerics

q = aqt_quantizer.Quantizer(
    numerics=int_numerics.IntSymmetric(
        bits=4,
        preserve_zero=True,
        preserve_max_val=True,
        clip=True,
        clip_gradient=True,
        round=True,
        noise_fn=None,
    ),
    calib_shared_axes=-1,
    scale_stop_grad=True,
    calibration=calibration.AbsMaxCalibration(),
    po2_scale=False,
    context=aqt_utils.Context(key=jax.random.PRNGKey(0), train_step=0))
```

To view the quantized weights created by the quantizer, try quantizing a simple vector like so. Remember that these are the quantized values that are stored in memory for weight quantization.

```python
x = jnp.linspace(-10, 10, 10)
x_q, _ = q.quant(x, calibration_axes=-1)
print(x_q.qvalue)
```

To view the dequantized values that are used after multiplying the quantized values by the scale, use this code

```python
print(x_q.dequant())
```

To view the scales, simply call the scale attribute

```python
print(x_q.scale)
```

For a sanity check, we can assert that the scale in this example is simply equal to the dequantized values divided by the quantized values. (Hint: if you get an error check for nan values)

```python
assert jnp.all(x_q.dequant() / x_q.qvalue == x_q.scale[0])
```

Finally, we can plot the quantized values like so

```python
import matplotlib.pyplot as plt
plt.plot(x, x_q.qvalue)
```

The specific quantizer parameters here are implemented in this tutorial are just for demonstration purposes and can be easily changed. Try altering the number of bits and see how the number of quantization steps changes accordingly.


## AQT Versions

As of today there are several independent AQT implementations in this package:

- [JAX Legacy AQT](https://github.com/google/aqt/blob/main/aqt/jax_legacy)
  Obsolete version of AQT still used by some customers.
- [JAX AQTv1](https://github.com/google/aqt/blob/main/aqt/jax)
  Version of AQT that was developed with acceleration of NN inference in mind.
- [TF AQTv1](https://github.com/google/aqt/blob/main/aqt/tensorflow)
  Tensorflow counterpart of JAX AQTv1.
- [JAX AQTv2](https://github.com/google/aqt/blob/main/aqt/jax/v2)
  AQT implementing universal matmul quantization.

AQTv2 is the recommended library.
We plan to port remaining features from AQTv1 to AQTv2 and
delete AQTv1 in early Q1 2024. Below we describe details about that.

## Inference Acceleration

The most important AQTv2 (to be ported from AQTv1) missing features are:

 - https://github.com/google/aqt/issues/282
 - https://github.com/google/aqt/issues/280


Lack of these features prevents AQTv2 from accelerating inference with small batch.
The only option today is dynamic quantization where
each tensor op is quantized independently and quantization scales are found just-in-time.

## Backpropagation Acceleration

AQTv2 speeds up training and fine-tuning.
We verified 1.2x to 1.4x reduction in step time on 1B to 16B large Transformer
models to a given quality on TPUs.

Today in order to do it correctly one needs to understand that for each
two-argument tensor op (matmul, einsum, conv) in the forward pass,
there are two in the backward pass.
One has to understand how to configure them.

We will be updating config file with current best practices.

## AQT Serving

Quantization is applied to both activations and weights right before a matmul.
During training, gradients flow "through" the operation and update the latent
floating-point weights.
During model serving, however, re-quantizing weights is unnecessary and has large overhead because:

1. Weights are stationary during serving. Recomputing weight quantization is a waste.
2. Recomputing quantization requires loading the latent BF16 weights, which
consumes 2x more memory bandwidth than loading INT8 weights.

AQT provides a solution to the above problem, which is called "serving conversion".
Serving conversion is a constant folding process that creates new variables in
the checkpoint to store the folded INT8 weights.
It requires running one dummy inference.

Note that:

1. Weights can be either lhs or rhs inputs to a matmul. AQT supports storing
both in checkpoints, configured by [lhs_quant_mode and rhs_quant_mode](https://github.com/google/aqt/blob/eb445f8fd8cefac33eb965f0115ea2ac3d5cdff5/aqt/jax/v2/flax/aqt_flax.py#L97-L105).
Users need to set the correct configuration themselves.
2. In some cases where checkpoint variables require metadata such as sharding axis,
users can configure [variable initializers](https://github.com/google/aqt/blob/eb445f8fd8cefac33eb965f0115ea2ac3d5cdff5/aqt/jax/v2/flax/aqt_flax.py#L98-L99) to fit their needs.

At serving mode, AQT will look for the INT8 variables in a checkpoint.
If found, it skips the weight quantization and returns INT8 variables
as-is as inputs to matmul, thus saving the memory bandwidth and avoids recalculation.

Applying the conversion to an unquantized floating-point model is
equivalent to post-training quantization (PTQ) serving.
Applying the conversion to a forward-only quantized AQT model is
quantization-aware training (QAT) serving.
In that case it is important to use the same AQT config during training and serving to maintain WYTIWYS.

The [flax end-to-end example](https://github.com/google/aqt/blob/4c3f3d915a84089ff62a5c6cddf007949ed6fd96/aqt/jax/v2/examples/flax_e2e_model.py#L214-L235)
provides a code snippet on how to perform serving conversion and model serving in AQT.

## Other Weight Transformations

Consider `matmul(a, w)` as an activation-weight matmul.
Sometimes there is another transformation `T` on weights before passed into the matmul,
i.e., `matmul(..., T(w))` is computed.
In this case, quantizing weights directly, i.e., `matmul(..., T(Q(w)))`, can reduce the
checkpoint size and save memory bandwidth, but it will not accelerate the matmul
because `T` will likely return floats.
In order to get matmul acceleration, the quantization function `Q` should be
inserted just before matmul, i.e., `matmul(..., Q(T(w)))`.

AQT pursues the goal of both compressing the checkpoint AND accelerating the matmul.
This requires storing the entire `w_q = Q(T(w))` in the checkpoint and
using it in serving directly, i.e., `matmul(..., w_q)`.

Note that AQT provides a quantized `matmul_aqt`
as a whole such that `matmul_aqt(..., T(w)) = matmul(..., Q(T(w)))`.
`Q(T(w))` is not visible outside of `matmul_aqt`.
The main reason is that `matmul_aqt` has custom gradient defined for it.

## How AQT Works Internally

In this section we:

  * show how to get quantization acceleration in JAX,
  * explain what AQT INT8 does under-the-hood (using the simplest INT8 configuration),
  * run the code on a simple example.

Code in this section can be found and executable in the [example colab](https://github.com/google/aqt/blob/main/aqt/jax/v2/examples/examples.ipynb).
Note that this section mainly explains how AQT works and why it can achieve a good quality. For AQT tutorial, user can refer to the [usage](#usage) section.

The `matmul_true_int8` takes real INT8 as inputs, returns int32. The matmul computation `jnp.matmul` calls [lax.dot_general](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.dot_general.html) in its source, which is a JAX wrapper for [XLA DotGeneral](https://www.tensorflow.org/xla/operation_semantics#dotgeneral) op that implements all MXU ops (this is where we have int8 acceleration on TPUs) except convolution. This is how one can get hardware acceleration of quantized matmul in JAX.

```python
import jax.numpy as jnp

def matmul_true_int8(lhs, rhs):
  assert lhs.dtype == jnp.int8
  assert rhs.dtype == jnp.int8
  result = jnp.matmul(lhs, rhs, preferred_element_type=jnp.int32)
  assert result.dtype == jnp.int32
  return result
```

Generate some random data:

```python
batch_size = 3
channels_in = 4
channels_out = 5
a = gen_matrix(batch_size, channels_in) # Activations
w = gen_matrix(channels_in, channels_out) # Weights
```

Below is how AQT works internally using the simplest INT8 configuration.
Even though names such as "batch" and "channels" are used, "w" and "a", which are evocative of neural networks, one may note that `aqt_matmul_int8` algorithm is not DNN specific.

```python
def aqt_matmul_int8(a, w):
  max_int8 = 127
  # This function is customizable and injectable, i.e:
  # users can inject custom quant code into an AQT config.
  def quant_int8(x):
    return jnp.clip(jnp.round(x), -max_int8, max_int8).astype(jnp.int8)

  # Calibration. Calibration function is also customizable and injectable.
  a_s = max_int8 / jnp.max(jnp.abs(a), axis=1, keepdims=True)
  w_s = max_int8 / jnp.max(jnp.abs(w), axis=0, keepdims=True)
  assert a_s.shape == (batch_size, 1) # shapes checked for illustration
  assert w_s.shape == (1, channels_out)

  # int8 matmul with int32 accumulator
  result = matmul_true_int8(quant_int8(a * a_s), quant_int8(w * w_s)) / (a_s * w_s)
  assert result.shape == (batch_size, channels_out)

  return result
```

Note that each example in a batch and each output channel will have their own separate scale. This reduces the effect of outliers in "w" and "a" to just one row or column, making a tighter calibration and much better quality of quantization. Comparing aqt_matmul_int8 to float matmul, their outputs are close.

```python
print(f"jnp.matmul(a, w):\n", jnp.matmul(a, w))
print(f"aqt_matmul_int8(a, w):\n", aqt_matmul_int8(a, w))
```

```
# should expect the following outputs
jnp.matmul(a, w):
 [[ 3.6095254   5.8575077   1.9510972   4.732388    1.9792626 ]
 [ 4.335892    0.9743651   2.7298734   4.3540883   3.637487  ]
 [-0.07735002  2.7310796  -0.3519049   0.19912864 -1.2023292 ]]
aqt_matmul_int8(a, w):
 [[ 3.5998788   5.8562713   1.9385538   4.7426414   1.9792401 ]
 [ 4.321886    0.99681264  2.737299    4.3591022   3.6352503 ]
 [-0.07714217  2.7415617  -0.35343346  0.20568734 -1.1974115 ]]
```

## Citing AQT
We will be publishing AQT whitepaper soon.
