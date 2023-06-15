# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for dot_general."""
import copy
from absl.testing import absltest
from absl.testing import parameterized
from aqt.jax.v2 import config
import aqt.jax.v2.aqt_dot_general as aqt
import flax.linen.linear as fl
import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats


def test_jaxpr(f, cfgs: list[config.DotGeneralRaw]):
  """Tests whether dot_generals in f conform to cfgs."""

  def jaxpr_to_trityp(jaxpr):
    for eq in jaxpr.eqns:
      # This is where dot_generals hide when one uses custom vjp
      if "fun_jaxpr" in eq.params.keys():
        for rec in jaxpr_to_trityp(eq.params["fun_jaxpr"]):
          yield rec

      if eq.primitive.name == "dot_general":
        [lhs, rhs] = eq.invars
        [out] = eq.outvars
        trityp = (lhs.aval, rhs.aval, out.aval)
        yield trityp

  f_jaxpr = jax.make_jaxpr(f)()
  trityps = [trityp for trityp in jaxpr_to_trityp(f_jaxpr)]
  assert len(trityps) == len(cfgs)
  for (lhs_sa, rhs_sa, out_sa), cfg in zip(trityps, cfgs):
    assert lhs_sa.dtype == cfg.lax_dg_in_dtype
    assert rhs_sa.dtype == cfg.lax_dg_in_dtype
    assert out_sa.dtype == cfg.lax_dg_out_dtype


def rand_unif(shape, maxval, seed):
  key = jax.random.PRNGKey(seed)
  return jax.random.uniform(key=key, shape=shape, minval=-maxval, maxval=maxval)


# The main test strategy is :
# - Test that FQ is sensible (improve fq_noise_test)
#   - Quantization noise (rounding error) is of proper value
#   - Cliping is correct
#   - Stochastic noise is added correctly.
# - Test that we really use int8/int4/bool in XLA/HLO
# - Test of fq vs dot_general equivalence in po2
# - Test that it all applies to gradients
#
#
# TODO(lew): Tests are a bit incomplete. What we need:
# On top of that we shell test:
#  - Gradinent is 1 in the range, 0 outside the abs-max range.


def test_eq(name, a, b):
  # TODO(lew): use library function.
  mean_err = jnp.mean(jnp.abs(a - b))
  if mean_err != 0.0:
    print("mean_err =", mean_err)
    print(a.shape)
    print(a[:3, :3])
    print(b.shape)
    print(b[:3, :3])
    print(f"FAIL: {name}")
    assert False


def fqt_param_dict(s):
  return dict(
      cfg=config.fully_quantized(8, True),
      seed=s,
  )


class AqtDotGeneralResearchTest(parameterized.TestCase):

  def test_empty(self):
    t = np.random.normal(size=6).reshape((2, 3))
    np.testing.assert_array_equal(t, t)

  def test_fq_noise(self):
    for preserve_zero in [True, False]:
      for prec in [1, 2, 4, 8]:
        for v in [0.1, 1000.0]:
          for seed in range(10):
            key = jax.random.PRNGKey(seed)
            cfg = config.Tensor.make(prec)
            cfg.numerics.preserve_zero = preserve_zero
            cfg.calib_shared_axes = (0,)
            sample_size = 10000
            shape = (sample_size,)
            a = jax.random.uniform(key, shape, minval=-v, maxval=v)
            context = aqt.Context(key=None, train_step=None)
            a_fq = aqt.make_fake_quant(cfg)(a, context)
            bucket_noise = a_fq - a  #  ~ U(-bucket_size/2, bucket_size/2)
            bucket_count = (2**prec - 1) if preserve_zero else (2**prec)
            bucket_size = (v * 2) / bucket_count
            noise = bucket_noise / bucket_size + 0.5  # ~U(0, 1)
            pvalue = scipy.stats.kstest(noise, "uniform").pvalue
            assert pvalue > 0.01

  @parameterized.parameters([
      dict(bits=1),
  ])
  def test_fake_quant(
      self,
      bits=4,
      maxval=10.0,
      shape=(20, 1),
  ):
    cfg = config.Tensor.make(bits)
    cfg.po2_scale = True
    cfg.calib_shared_axes = (0,)
    x = jnp.linspace(-maxval, maxval, num=shape[0]).reshape(shape)
    grad = jnp.ones(shape) * 12345.0
    context = aqt.Context(key=None, train_step=None)
    x_fq, backprop = jax.vjp(aqt.make_fake_quant(cfg), x, context)
    gx_fq = backprop(grad)
    # print(f"x     =\n{x}")
    # print(f"x_fq  =\n{x_fq}")
    # print(f"grad  =\n{grad}")
    # print(f"gx_fq =\n{gx_fq}")
    del x
    del x_fq
    del grad
    del gx_fq

    # TODO(lew): test

  @parameterized.parameters([
      *[dict(cfg=config.fully_quantized(8, False), seed=s) for s in range(10)],
      *[fqt_param_dict(s) for s in range(10)],
      dict(cfg=config.DotGeneral.make(None, None)),
      dict(cfg=config.DotGeneral.make(1, 1)),
      dict(cfg=config.DotGeneral.make(1, 2)),
      dict(cfg=config.DotGeneral.make(2, 1)),
      dict(cfg=config.DotGeneral.make(2, 2)),
      dict(cfg=config.DotGeneral.make(8, 8)),
      dict(cfg=config.DotGeneral.make(None, 8)),
      dict(cfg=config.DotGeneral.make(8, None)),
      dict(
          cfg=fqt_param_dict(s=10)["cfg"],
          dims=(((0, 2), (1, 0)), ((3, 1), (2, 4))),
          # contraction: 2, 5; batch: 4, 3
          lhs_shape=(2, 3, 5, 4),  # non-contr: 3, 4
          rhs_shape=(5, 2, 4, 6, 3),  # non-contr: 4, 6, 3
          gra_shape=(4, 3, 6),
      ),
      dict(
          cfg=config.DotGeneral.make(2, 2),
          dims=(((0, 2), (1, 0)), ((3, 1), (2, 4))),
          # contraction: 2, 5; batch: 4, 3
          lhs_shape=(2, 3, 5, 4),  # non-contr: 3, 4
          rhs_shape=(5, 2, 4, 6, 3),  # non-contr: 4, 6, 3
          gra_shape=(4, 3, 6),
      ),
  ])
  def test_dot_general(
      self,
      cfg: config.DotGeneral,
      lhs_maxval=10.0,
      rhs_maxval=20.0,
      gra_maxval=30.0,
      dims=(((1,), (0,)), ((), ())),  # classical matmul
      # lhs_shape=(1, 1),
      # rhs_shape=(1, 2),
      # gra_shape=(1, 2),  # has to be the shape of the output
      # lhs_shape=(1, 2),
      # rhs_shape=(2, 3),
      # gra_shape=(1, 3),  # has to be the shape of the output
      lhs_shape=(10, 20),
      rhs_shape=(20, 30),
      gra_shape=(10, 30),  # has to be the shape of the output
      seed=0,
  ):
    readonly_cfg = cfg
    del cfg

    def modify_cfg(use_fake_quant=False, *, use_fwd_quant=None):
      cfg = copy.deepcopy(readonly_cfg)
      # Setting po2_scale is ensuring that fake_quant and full dot_general
      # have the same numerics when scales are power of two (po2).
      # We are passing dims to config so that we can reuse it in fake_quant.
      # Power-of-2 scales allow FQ and AQT to be exactly the same.
      cfg.fwd.lhs.po2_scale = True
      cfg.fwd.rhs.po2_scale = True
      cfg.dlhs.lhs.po2_scale = True
      cfg.dlhs.rhs.po2_scale = True
      cfg.drhs.lhs.po2_scale = True
      cfg.drhs.rhs.po2_scale = True
      cfg.fwd.use_fake_quant = use_fake_quant
      cfg.dlhs.use_fake_quant = use_fake_quant
      cfg.drhs.use_fake_quant = use_fake_quant

      def disable_quant_types(c):
        c.lax_dg_in_dtype = jnp.bfloat16
        c.lax_dg_out_dtype = jnp.float32

      if use_fake_quant:
        disable_quant_types(cfg.fwd)
        disable_quant_types(cfg.dlhs)
        disable_quant_types(cfg.drhs)

      if use_fwd_quant is not None:
        # If we disable all rounding, we are just testing whether the scales are
        # correct. We don't even need to disable clipping and we are testing
        # that the scales are not too large.
        def disable_quant(c):
          disable_quant_types(c)
          c.lhs.round = False
          c.rhs.round = False
          # c.lhs.clip = False
          # c.rhs.clip = False
          c.lax_dg_in_dtype = jnp.bfloat16
          c.lax_dg_out_dtype = jnp.float32
          c.use_fwd_quant = use_fwd_quant
        disable_quant(cfg.fwd)
        disable_quant(cfg.dlhs)
        disable_quant(cfg.drhs)
      return cfg

    # test dot_general
    lhs = rand_unif(lhs_shape, lhs_maxval, seed)
    rhs = rand_unif(rhs_shape, rhs_maxval, seed + 1)
    gra = rand_unif(gra_shape, gra_maxval, seed + 2)

    def aqt_dg_full(use_fake_quant, use_fwd_quant=None):
      cfg = modify_cfg(use_fake_quant, use_fwd_quant=use_fwd_quant)
      dg = aqt.make_dot_general(cfg)
      context = aqt.Context(key=None, train_step=None)
      return lambda lhs, rhs: dg(lhs, rhs, dims, context=context)

    def aqt_dg_raw(use_fake_quant):
      cfg = modify_cfg(use_fake_quant).fwd
      dg_raw = aqt._make_dot_general_raw(cfg)
      context = aqt.Context(key=None, train_step=None)
      return lambda lhs, rhs: dg_raw(lhs, rhs, dims, context)[0]

    # Test that with backprop correctly composes 3 functions.
    # We need to test shape calculations and the returned values.
    # For the former we have multiple shapes,
    # for the latter we add some constant and test it on return.
    def lax_dg(lhs, rhs):
      return jax.lax.dot_general(lhs, rhs, dims)

    def lax_dg_248(lhs, rhs):
      def dg_mul(delta):

        def dg(
            lhs,
            rhs,
            dimension_numbers,
            context,
        ):
          ret = jax.lax.dot_general(lhs, rhs, dimension_numbers)
          ret *= delta

          def res(v):
            return aqt.TensorRes(
                value=v, qvalue=v, qvalue_scale=None, quant_grad=None
            )

          res = aqt.DotGeneralRes(
              context_bwd=context,
              lhs=res(lhs),
              rhs=res(rhs),
          )
          return ret, res

        return dg

      m1 = dg_mul(2.0)
      m2 = dg_mul(4.0)
      m3 = dg_mul(8.0)
      return aqt._dot_general_raw_attach_gradient(m1, m2, m3)(
          lhs, rhs, dims, context=aqt.Context(key=None, train_step=None)
      )

    # Test whether dtypes are correct in jaxpr
    test_jaxpr(lambda: aqt_dg_full(False)(lhs, rhs), [modify_cfg().fwd])
    test_jaxpr(
        lambda: jax.vjp(aqt_dg_full(False), lhs, rhs), [modify_cfg().fwd]
    )
    _, backprop = jax.vjp(aqt_dg_full(False), lhs, rhs)
    test_jaxpr(lambda: backprop(gra), [modify_cfg().dlhs, modify_cfg().drhs])

    # Test exact equalities.
    def check(dgs):
      good_lr = None
      good_gl = None
      good_gr = None
      for name, dg, options in dgs:
        # Default settings:
        if "test_gradient" not in options:
          options["test_gradient"] = True
        if "mult" not in options:
          options["mult"] = (1.0, 1.0, 1.0)

        lrf = dg(lhs, rhs)
        lr, backprop = jax.vjp(dg, lhs, rhs)

        gl, gr = backprop(gra) if options["test_gradient"] else (None, None)
        good_lr = lr if good_lr is None else good_lr
        good_gl = gl if good_gl is None else good_gl
        good_gr = gr if good_gr is None else good_gr

        test_eq(f"{name}: lr vs lrf", lr, lrf)

        lr_mult, gl_mult, gr_mult = options["mult"]
        test_eq(f"{name}: lr", good_lr, lr / lr_mult)  # forward pass
        if options["test_gradient"]:
          test_eq(f"{name}: gl", good_gl, gl / gl_mult)  # backward pass
          test_eq(f"{name}: gr", good_gr, gr / gr_mult)  # backward pass

    check([
        ("default    ", aqt_dg_full(False), dict()),
        ("FQ         ", aqt_dg_full(True), dict()),
        ("raw fwd    ", aqt_dg_raw(False), dict(test_gradient=False)),
        ("raw fwd FQ ", aqt_dg_raw(True), dict(test_gradient=False)),
    ])

    check([
        ("fwd_quant=T", aqt_dg_full(False, use_fwd_quant=False), dict()),
        ("fwd_quant=F", aqt_dg_full(False, use_fwd_quant=True), dict()),
    ])

    check([
        ("lax_dg    ", lax_dg, dict()),
        ("lax_dg_248", lax_dg_248, dict(mult=(2.0, 4.0, 8.0))),
    ])

  def test_dynamic_context(self):
    @jax.jit
    def f(lhs, rhs, context):
      cfg = config.DotGeneral.make()
      dg = aqt.make_dot_general(cfg)
      return dg(lhs, rhs, (((0,), (0,)), ((), ())), context=context)

    lhs, rhs = jnp.array([3.0, 4.0]), jnp.array([4.0, 5.0])
    context = aqt.Context(
        key=jax.random.PRNGKey(4), train_step=None
    )  # xkcd.com/221
    jax.value_and_grad(f)(lhs, rhs, context)

  def test_hardware_int8(self, seed=0):
    cfg = config.DotGeneralRaw.make(8, 8)

    def dg(lhs, rhs):
      ret, _ = aqt._make_dot_general_raw(cfg)(
          lhs,
          rhs,
          (((1,), (0,)), ((), ())),
          aqt.Context(key=None, train_step=None),
      )
      return ret

    lhs = rand_unif((10, 20), 1.0, seed)
    rhs = rand_unif((20, 30), 1.0, seed + 1)
    test_jaxpr(lambda: dg(lhs, rhs), [cfg])
    assert cfg.lax_dg_in_dtype == jnp.int8
    assert cfg.lax_dg_out_dtype == jnp.int32

  @parameterized.parameters([
      (1, 1),
      (1, 2),
      (2, 1),
      (2, 2),
      (8, 8),
      (None, 8),
      (8, None),
      (None, None),
  ])
  def test_conv_general_dilated(
      self,
      lhs_bits,
      rhs_bits,
      lhs_maxval=10.0,
      rhs_maxval=20.0,
      seed=0,
  ):
    cfg = config.DotGeneralRaw.make_conv_general_dilated(2, lhs_bits, rhs_bits)

    if cfg.lhs:
      # Power-of-2 scales allow FQ and AQT to be exactly the same.
      cfg.lhs.po2_scale = True
    if cfg.rhs:
      cfg.rhs.po2_scale = True

    batch_n = 10
    contr_n = 20
    feature_n = 30
    lhs = rand_unif((batch_n, 4, 5, contr_n), lhs_maxval, seed)
    rhs = rand_unif((3, 3, contr_n, feature_n), rhs_maxval, seed + 1)

    lax_conv = jax.lax.conv_general_dilated
    aqt_conv = aqt.make_conv_general_dilated(cfg)
    kwargs = {
        "window_strides": (1, 1),
        "padding": "SAME",
        "dimension_numbers": fl._conv_dimension_numbers(lhs.shape),
    }
    lhs_fq = aqt.make_fake_quant(cfg.lhs)(
        lhs, aqt.Context(key=None, train_step=None)
    )
    rhs_fq = aqt.make_fake_quant(cfg.rhs)(
        rhs, aqt.Context(key=None, train_step=None)
    )
    prod_fq = lax_conv(lhs_fq, rhs_fq, **kwargs)
    prod_aqt = aqt_conv(lhs, rhs, **kwargs)
    assert (prod_aqt == prod_fq).all()


if __name__ == "__main__":
  absltest.main()
