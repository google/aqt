"""Flax implementation of PokeBNN."""

# pylint: disable=unnecessary-lambda
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=using-constant-test

from typing import Any

from flax import linen as nn
import jax.numpy as jnp


# from aqt.jax import flax_layers as aqt_flax_layers
# from aqt.jax import quant_config
from google3.third_party.google_research.google_research.aqt.jax import flax_layers as aqt_flax_layers
from google3.third_party.google_research.google_research.aqt.jax import quant_config


class DPReLU(nn.Module):
  """Implements: dprelu(x) = slope * (x - bias_x) - bias_y.

     where slope depends on the sign of (x - bias_x).
  """

  @nn.compact
  def __call__(self, x):
    params_shape = (x.shape[-1],)

    bias_init_fn = lambda key, shape: jnp.ones(shape) * 0.0
    bias_x = self.param('bias_x', bias_init_fn, params_shape)
    bias_y = self.param('bias_y', bias_init_fn, params_shape)

    neg_slope_init_fn = lambda key, shape: jnp.ones(shape) * 0.25
    pos_slope_init_fn = lambda key, shape: jnp.ones(shape)
    neg_slope = self.param('neg_slope', neg_slope_init_fn, params_shape)
    pos_slope = self.param('pos_slope', pos_slope_init_fn, params_shape)

    x = x - bias_x
    slope = jnp.where(x >= 0, pos_slope, neg_slope)
    x = slope * x - bias_y
    return jnp.asarray(x, jnp.bfloat16)


class PokeBNN(nn.Module):
  # All the configurations goes through hparams dictionary.
  num_classes: int
  hparams: Any
  # Dynamic information on whether we should be quantizing already.
  quant_context: quant_config.QuantContext
  train: bool
  # Jax way of multi-machine distribution
  paxis_name: Any
  # Logs all the ops with computational contents. Used for ACE calculation.
  op_log = []

  # Instruments other ops to log shaps and their arguments to self.op_log.
  def instr(self, name, op, other):
    def f(x, **kwargs):
      y = op(x, **kwargs)
      self.op_log.append((name, x.shape, y.shape, other))
      return y
    return f

  # All the common setting, plumbing and instrumentation.
  def batch_norm(self, **kwargs):
    op = nn.BatchNorm(
        use_running_average=not self.train,
        momentum=0.9,
        epsilon=1e-5,
        dtype=jnp.bfloat16,
        **kwargs)
    return self.instr('batch_norm', op, kwargs)

  # All the common setting, plumbing and instrumentation.
  def conv(self, use_bias=False, **kwargs):
    # ConvAqt implements quantization calibration for 8-, 4- and 1-bit convs.
    op = aqt_flax_layers.ConvAqt(
        use_bias=use_bias,
        dtype=jnp.bfloat16,
        quant_context=self.quant_context,
        paxis_name=self.paxis_name,
        train=self.train,
        **kwargs)
    return self.instr('conv', op, kwargs)

  # All the common setting, plumbing and instrumentation.
  def dense(self, **kwargs):
    op = aqt_flax_layers.DenseAqt(
        dtype=jnp.bfloat16,
        train=self.train,
        quant_context=self.quant_context,
        paxis_name=self.paxis_name,
        hparams=self.hparams.dense_layer,
        **kwargs)
    return self.instr('dense', op, (kwargs, self.hparams.dense_layer))

  # Instrumentation of DPReLU.
  def dprelu(self):
    op = DPReLU()
    return self.instr('dprelu', op, ())

  # reshape_add as described in the paper + some manual detailed instrumentation
  def reshape_add(self, r, yy, method):
    """Match the dimension for residuals and add to activations."""
    if r is None: return yy
    out_features = yy.shape[3]
    in_features = r.shape[-1]
    if in_features > out_features:
      num_ch_avg = in_features // out_features
      r = r[:, :, :, 0:out_features * num_ch_avg]
      dim_nwh = r.shape[0:3]
      r = jnp.reshape(r, dim_nwh + (out_features, num_ch_avg))
      self.op_log.append(('reshape_add:average axis=4; before', r.shape))
      r = jnp.average(r, axis=4)
    elif in_features < out_features:
      ch_mult = out_features // in_features
      assert method in ['tile', 'zeropad']
      if method == 'tile':
        self.op_log.append(
            ('reshape_add:tile axis=3; before', r.shape, ch_mult))
        r = jnp.tile(r, reps=(1, 1, 1, ch_mult))
      if method == 'zeropad':
        pad_size = out_features - in_features
        pad_width = ((0, 0), (0, 0), (0, 0), (0, pad_size))
        self.op_log.append(
            ('reshape_add:zeropad axis=3; before', r.shape, pad_size))
        r = jnp.pad(r, pad_width=pad_width, mode='constant')
    if yy.shape != r.shape:
      r = nn.avg_pool(r, window_shape=(3, 3), padding='SAME', strides=(2, 2))
      self.op_log.append(('reshape_add:avg_pool3x3, st=2x2; after', r.shape))
    self.op_log.append(('reshape_add:add ' + method, r.shape))
    return yy + r

  def poke_init(self, x):
    # No input pixel is used twice.
    x = self.conv(
        features=self.hparams.init_group,
        kernel_size=(4, 4),
        strides=(4, 4),
        padding='VALID',
        name='init_conv',
        hparams=self.hparams.conv_init)(x)
    x = self.batch_norm(name='init_bn')(x)
    x = self.dprelu()(x)
    x = self.conv(
        features=64,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='SAME',
        feature_group_count=self.hparams.init_group,
        name='init_conv_group',
        hparams=self.hparams.conv_init)(x)
    x = self.batch_norm(name='init_bn_group')(x)
    x = self.dprelu()(x)
    return x

  def se(self, x, features, name):
    proj_len = x.shape[-1] // 8
    hparams = self.hparams.residual_blocks[0].conv_se
    self.op_log.append(('se:global_avg_pool axis = 1,2; before', x.shape))
    # TODO(yichi): Test that this line replaces the next two and do it.
    # s = jnp.mean(x, axis=(1, 2), keepdims=True)
    s = jnp.apply_over_axes(jnp.mean, x, (1, 2))
    s = s.reshape(-1, 1, 1, x.shape[-1])

    hparams.quant_act.input_distribution = 'symmetric'  # use int4
    s = self.conv(
        use_bias=True,
        kernel_size=(1, 1),
        features=proj_len,
        hparams=hparams,
        name='sefc1'+name)(s)
    s = nn.relu(s)
    self.op_log.append(('se:relu', s.shape))

    hparams.quant_act.input_distribution = 'positive'  # use uint4, save 1 bit
    s = self.conv(
        use_bias=True,
        kernel_size=(1, 1),
        features=features,
        hparams=hparams,
        name='sefc2'+name)(s)
    s = jnp.minimum(jnp.maximum(s+3, 0), 6.) / 6
    self.op_log.append(('se:relu6 shifted', s.shape))

    return s

  def poke_conv(self, x, r1, name, features, kernel_size, strides):
    r = x
    x = self.conv(
        features=features,
        kernel_size=kernel_size,
        name='conv' + name,
        strides=strides,
        hparams=self.hparams.residual_blocks[0].conv_1)(x)
    # ResNet does this. It is important.
    scale_init = nn.initializers.zeros if name == '3' else nn.initializers.ones
    x = self.batch_norm(name='bn' + name, scale_init=scale_init)(x)
    x = self.reshape_add(r, x, 'zeropad')
    x = self.reshape_add(r1, x, 'tile')
    x = self.dprelu()(x)
    s = self.se(x=r, features=x.shape[3], name='se11' + name)
    x = x * s
    self.op_log.append(('se:poke_relu scale', x.shape, s.shape))

    x = self.batch_norm(name='bn_split' + name)(x)
    return x

  def bottleneck_block(self, x, name, hparams, features, strides):
    r1 = x
    x = self.poke_conv(
        x,
        r1=None,
        name=name + '1',
        features=features,
        kernel_size=(1, 1),
        strides=(1, 1),
    )
    x = self.poke_conv(
        x,
        r1=None,
        name=name + '2',
        features=features,
        kernel_size=(3, 3),
        strides=strides,
    )
    x = self.poke_conv(
        x,
        r1=r1,
        name=name + '3',
        features=features * 4,
        kernel_size=(1, 1),
        strides=(1, 1),
    )
    return x

  @nn.compact
  def __call__(self, x):
    self.op_log[:] = []
    # self.ace[0] = 0

    x = self.poke_init(x)

    # This list was generated from ResNet-50, and probably can be improved.
    for i, strides, features in [
        (0, (1, 1), 64),
        (1, (1, 1), 64),
        (2, (1, 1), 64),
        (3, (2, 2), 128),
        (4, (1, 1), 128),
        (5, (1, 1), 128),
        (6, (1, 1), 128),
        (7, (2, 2), 256),
        (8, (1, 1), 256),
        (9, (1, 1), 256),
        (10, (1, 1), 256),
        (11, (1, 1), 256),
        (12, (1, 1), 256),
        (13, (2, 2), 512),
        (14, (1, 1), 512),
        (15, (1, 1), 512),
    ]:
      x = self.bottleneck_block(
          x,
          name=f'resblock{i}',
          features=int(features * self.hparams.filter_multiplier),
          hparams=self.hparams.residual_blocks[i],
          strides=strides,
      )

    self.op_log.append(('pokebnn: global mean axis=1,2; before', x.shape))
    x = jnp.mean(x, axis=(1, 2))

    x = self.dense(features=self.num_classes)(x, padding_mask=None)
    x = jnp.asarray(x, jnp.bfloat16)
    return x

#   # This function calculates all the stats for the 'elementwise' appendix.
#   def stats(self):
#     conv_ace = 0
#     dense_ace = 0
#     batch_norm_sops = 0  # sops means scalar ops (per pixel, per feature)
#     bprelu_sops = 0
#     reshape_add_ch_shrink_sops = 0
#     reshape_add_ch_shrink_mul_sops = 0
#     reshape_add_avg_pool_sops = 0
#     reshape_add_block_residual_sops = 0
#     reshape_add_local_residual_sops = 0
#     se_global_pool_add_sops = 0
#     se_global_pool_mul_sops = 0
#     se_relu_sops = 0
#     se_relu6_sops = 0
#     se_mul_sops = 0
#     pokebnn_global_pool_sops = 0

#     # mem_read_bits = 0
#     # mem_write_bits = 0

#     for op in self.op_log:
#       name = op[0]
#       if False: pass

#       elif name == 'conv':
#         _, xs, ys, kwargs = op
#         out_pixels = ys[1] * ys[2]
#         ks = kwargs['kernel_size']
#         # fgc takes into account depth-wise layer
#         fgc = kwargs.get('feature_group_count', 1)
#         dot_length = ks[0] * ks[1] * (xs[3] / fgc)
#         adders =kwargs['hparams'].weight_prec*kwargs['hparams'].quant_act.prec
#         conv_ace += out_pixels * dot_length * ys[3] * adders

#       elif name == 'dense':
#         _, xs, ys, (kwargs, hparams) = op
#         adders = hparams.weight_prec * hparams.quant_act.prec
#         dense_ace += xs[1] * ys[1] * adders

#       elif name == 'dprelu':
#         _, xs, ys, _ = op
#         assert xs == ys
#         bprelu_sops += xs[1] * xs[2] * xs[3]

#       # During the training BN has two bf16 multiplications and two additions.
#       # In principle it can be folded to one of each.
#       # The energy would be: (210 + 110) fJ or 256 ACE per each sop.
#       elif name == 'batch_norm':
#         _, xs, ys, _ = op
#         assert xs == ys
#         batch_norm_sops += xs[1] * xs[2] * xs[3]

#       elif name == 'reshape_add:average axis=4; before':
#         _, sh = op
#         # height, width, #groups, group size
#         reshape_add_ch_shrink_sops += sh[1] * sh[2] * sh[3] * sh[4]
#         reshape_add_ch_shrink_mul_sops += sh[1] * sh[2] * sh[3]

#       elif name == 'reshape_add:tile axis=3; before':
#         # _, sh, ch_multiplier = op
#         pass

#       elif name == 'reshape_add:zeropad axis=3; before':
#         # _, sh, pad_size = op
#         pass

#       elif name == 'reshape_add:avg_pool3x3, st=2x2; after':
#         _, sh = op
#         reshape_add_avg_pool_sops += 3*3*sh[1]*sh[2]*sh[3]

#       elif name == 'reshape_add:add tile':
#         _, sh = op
#         reshape_add_block_residual_sops += sh[1]*sh[2]*sh[3]

#       # we don't need to add zeros during the computation on hw
#       elif name == 'reshape_add:add zeropad':
#         _, sh = op
#         reshape_add_local_residual_sops += sh[1]*sh[2]*sh[3]

#       elif name == 'se:global_avg_pool axis = 1,2; before':
#         _, sh = op
#         se_global_pool_add_sops += sh[1]*sh[2]*sh[3]
#         se_global_pool_mul_sops += sh[3]

#       elif name == 'se:relu':
#         _, sh = op
#         se_relu_sops += sh[1]*sh[2]*sh[3]

#       elif name == 'se:relu6 shifted':
#         _, sh = op
#         se_relu6_sops += sh[1]*sh[2]*sh[3]

#       elif name == 'se:poke_relu scale':
#         _, x_sh, _s_sh = op
#         # We can use s_sh (SE pre-pixel-broadcast) because in principle it can
#         # be fused with the following BN.
#         se_mul_sops += x_sh[1]*x_sh[2]*x_sh[3]

#       elif name == 'pokebnn: global mean axis=1,2; before':
#         _, sh = op
#         pokebnn_global_pool_sops += sh[1]*sh[2]*sh[3]

#       else: assert False

#     return {
#         'conv_ace': conv_ace,
#         'dense_ace': dense_ace,
#         'batch_norm_sops': batch_norm_sops,
#         'bprelu_sops': bprelu_sops,
#         'reshape_add_ch_shrink_sops': reshape_add_ch_shrink_sops,
#         'reshape_add_ch_shrink_mul_sops': reshape_add_ch_shrink_mul_sops,
#         'reshape_add_avg_pool_sops': reshape_add_avg_pool_sops,
#         'reshape_add_block_residual_sops': reshape_add_block_residual_sops,
#         'reshape_add_local_residual_sops': reshape_add_local_residual_sops,
#         'se_global_pool_add_sops': se_global_pool_add_sops,
#         'se_global_pool_mul_sops': se_global_pool_mul_sops,
#         'se_relu_sops': se_relu_sops,
#         'se_relu6_sops': se_relu6_sops,
#         'se_mul_sops': se_mul_sops,
#         'pokebnn_global_pool_sops': pokebnn_global_pool_sops,
#     }


def create_pokebnn(hparams, train, **kwargs):
  return PokeBNN(
      num_classes=1000,
      hparams=hparams,
      quant_context=quant_config.QuantContext(
          update_bounds=False, quantize_weights=True),
      train=train,
      paxis_name='batch',
      **kwargs)
