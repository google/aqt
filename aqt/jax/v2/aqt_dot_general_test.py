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
from aqt.jax.v2 import aqt_tensor
from aqt.jax.v2 import config
from aqt.jax.v2 import stochastic_rounding
import aqt.jax.v2.aqt_conv_general as aqt_conv
import aqt.jax.v2.aqt_dot_general as aqt
from aqt.jax.v2.numerics import int_numerics
from aqt.jax.v2.numerics import no_numerics
from aqt.jax.v2.numerics import numerics
import flax.linen.linear as fl
import flax.struct
import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats


def test_jaxpr_dtype(f, cfgs: list[config.DotGeneralRaw], float_dtype):
  """Tests whether dot_generals in f conform to dtypes inside of cfgs."""

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
    # If cfg has None, the type is inherited from the arguments' type.
    def assert_dtype_eq(dtype1, dtype2):
      assert dtype1 == dtype2, f"dtype1 != dtype2: {dtype1=} != {dtype2=}"

    assert_dtype_eq(lhs_sa.dtype, cfg.lhs.numerics.get_dtype() or float_dtype)
    assert_dtype_eq(rhs_sa.dtype, cfg.rhs.numerics.get_dtype() or float_dtype)
    assert_dtype_eq(out_sa.dtype, cfg.dg_accumulator_dtype or float_dtype)


def rand_unif(shape, maxval, seed, dtype=jnp.float32):
  key = jax.random.PRNGKey(seed)
  return jax.random.uniform(
      key=key, shape=shape, minval=-maxval, maxval=maxval, dtype=dtype
  )


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


def fqt_param_dict(s, use_fwd_quant, **kwargs):
  return dict(
      cfg=config.fully_quantized(
          use_fwd_quant=use_fwd_quant,
          use_stochastic_rounding=False,
          **kwargs,
      ),
      seed=s,
  )


class AqtDotGeneralResearchTest(parameterized.TestCase):

  def test_empty(self):
    t = np.random.normal(size=6).reshape((2, 3))
    np.testing.assert_array_equal(t, t)

  @parameterized.product(
      preserve_zero=[True, False],
      prec=[1, 2, 4, 8],
      v=[0.1, 1000.0],
      seed=list(range(10)),
  )
  def test_fq_noise(self, preserve_zero, prec, v, seed):
    key = jax.random.PRNGKey(seed)
    cfg = config.tensor_make(prec)
    if isinstance(cfg.numerics, int_numerics.IntNumerics):
      cfg.numerics = cfg.numerics.replace(preserve_zero=preserve_zero)
    cfg.calib_shared_axes = (0,)
    sample_size = 10000
    shape = (sample_size,)
    a = jax.random.uniform(key, shape, minval=-v, maxval=v)
    a_fq = aqt_tensor.make_fake_quant(cfg)(a)
    bucket_noise = a_fq - a  #  ~ U(-bucket_size/2, bucket_size/2)
    bucket_count = (2**prec - 1) if preserve_zero else (2**prec)
    bucket_size = (v * 2) / bucket_count
    noise = bucket_noise / bucket_size + 0.5  # ~U(0, 1)
    pvalue = scipy.stats.kstest(noise, "uniform").pvalue
    assert pvalue > 0.01

  def test_stochastic_rounding_noise(self):
    repeats = 1000
    num_values = 100000
    shape = (repeats, num_values)

    def assert_clt(noise: jnp.ndarray):
      # Test if the sample mean of the noise conforms to Central Limit Theorem

      # std of a uniform distribution [-0.5, 0.5]
      noise_std = 1 / jnp.sqrt(12.0)
      noise_mean = jnp.mean(noise, axis=1)
      assert jnp.all(noise_mean * jnp.sqrt(num_values) < 4 * noise_std)

    # jax.uniform implementation
    jax_uniform_noise = jax.random.uniform(jax.random.PRNGKey(10), shape) - 0.5
    assert_clt(jax_uniform_noise)
    # customized more efficient implementation
    custom_1_noise = stochastic_rounding.random_centered_uniform(
        shape, jax.random.PRNGKey(11)
    )
    assert_clt(custom_1_noise)

  @parameterized.parameters([
      dict(bits=1),
  ])
  def test_fake_quant(
      self,
      bits=4,
      maxval=10.0,
      shape=(20, 1),
  ):
    cfg = config.tensor_make(bits)
    cfg.po2_scale = True
    cfg.calib_shared_axes = (0,)
    x = jnp.linspace(-maxval, maxval, num=shape[0]).reshape(shape)
    grad = jnp.ones(shape) * 12345.0
    x_fq, backprop = jax.vjp(aqt_tensor.make_fake_quant(cfg), x)
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
      dict(
          cfg=config.config_v3(
              fwd_bits=3,
              dlhs_bits=4,
              drhs_bits=5,
              drhs_accumulator_dtype=jnp.int32,  # overwriting the default None
          )
      ),
      dict(cfg=config.dot_general_make(None, None)),
      dict(cfg=config.dot_general_make(1, 1)),
      dict(cfg=config.dot_general_make(1, 2)),
      dict(cfg=config.dot_general_make(2, 1)),
      dict(cfg=config.dot_general_make(2, 2)),
      dict(cfg=config.dot_general_make(8, 8)),
      dict(cfg=config.dot_general_make(8, 8), clip_gradient=True),
      dict(
          cfg=config.dot_general_make(8, 8, dlhs_local_aqt=config.LocalAqt(2))
      ),
      dict(
          cfg=config.dot_general_make(8, 8, drhs_local_aqt=config.LocalAqt(2))
      ),
      # That test could fail numerically because bf16
      # can't keep in the product of int8*int8 accurately.
      # It just so happens that this test does not fail but others do.
      # We do this test anyway, to catch jax-compilation-time errors.
      dict(cfg=config.dot_general_make(2, 2), dtype=jnp.bfloat16),
      dict(cfg=config.dot_general_make(8, 8), dtype=jnp.bfloat16),
      dict(cfg=config.dot_general_make(None, 8)),
      dict(cfg=config.dot_general_make(8, None)),
      dict(
          cfg=fqt_param_dict(s=10, use_fwd_quant=True)["cfg"],
          dims=(((0, 2), (1, 0)), ((3, 1), (2, 4))),
          # contraction: 2, 5; batch: 4, 3
          lhs_shape=(2, 3, 5, 4),  # non-contr: 3, 4
          rhs_shape=(5, 2, 4, 6, 3),  # non-contr: 4, 6, 3
          gra_shape=(4, 3, 6),
      ),
      dict(
          cfg=fqt_param_dict(
              s=10,
              use_fwd_quant=True,
              dlhs_local_aqt=config.LocalAqt(2),
          )["cfg"],
          dims=(((0, 2), (1, 0)), ((3, 1), (2, 4))),
          # contraction: 2, 5; batch: 4, 3
          lhs_shape=(2, 3, 5, 4),  # non-contr: 3, 4
          rhs_shape=(5, 2, 4, 6, 3),  # non-contr: 4, 6, 3
          gra_shape=(4, 3, 6),
      ),
      dict(
          cfg=config.dot_general_make(2, 2),
          dims=(((0, 2), (1, 0)), ((3, 1), (2, 4))),
          # contraction: 2, 5; batch: 4, 3
          lhs_shape=(2, 3, 5, 4),  # non-contr: 3, 4
          rhs_shape=(5, 2, 4, 6, 3),  # non-contr: 4, 6, 3
          gra_shape=(4, 3, 6),
      ),
      *[fqt_param_dict(s, use_fwd_quant=False) for s in range(10)],
      *[fqt_param_dict(s, use_fwd_quant=True) for s in range(10)],
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
      dtype=jnp.float32,
      clip_gradient=False,
  ):
    readonly_cfg = cfg
    del cfg

    def modify_cfg(
        *,
        use_fake_quant=False,
        use_fwd_quant=None,
        fwd_lhs_tricky_clip_and_round=False,
        local_aqt=None,
        clip_gradient=False,
    ):
      cfg = copy.deepcopy(readonly_cfg)
      if fwd_lhs_tricky_clip_and_round:
        # Tricky means that we have zero gradient on x < 0
        class TrickyNumerics(numerics.AqtNumerics, flax.struct.PyTreeNode):
          # Needed because int8 casting would do additional clip and round.
          dtype = None

          def get_dtype(self):
            return self.dtype

          def abs_val_mapped_to(self) -> jnp.ndarray:
            return jnp.array(1.0)

          def fwd(self, x, context):
            del context
            return jax.lax.round(x, jax.lax.RoundingMethod.TO_NEAREST_EVEN)
            # return x

          def vjp_fwd(self, x, context):
            res = (x,)
            return self.fwd(x, context), res

          def vjp_bwd(self, res, grad):
            (x,) = res
            ret = grad * (x >= 0)
            return (ret, None)

        cfg.fwd.lhs.numerics = TrickyNumerics()
        cfg.fwd.dg_accumulator_dtype = None

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

      cfg.fwd.lhs.use_fake_quant = use_fake_quant
      cfg.fwd.rhs.use_fake_quant = use_fake_quant

      cfg.dlhs.lhs.use_fake_quant = use_fake_quant
      cfg.dlhs.rhs.use_fake_quant = use_fake_quant

      cfg.drhs.lhs.use_fake_quant = use_fake_quant
      cfg.drhs.rhs.use_fake_quant = use_fake_quant

      def disable_quant_types(c):
        c.lhs.numerics = c.lhs.numerics.replace(dtype=None)
        c.rhs.numerics = c.rhs.numerics.replace(dtype=None)
        c.dg_accumulator_dtype = None

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
          if isinstance(c.lhs.numerics, int_numerics.IntNumerics):
            c.lhs.numerics = c.lhs.numerics.replace(round=False)
          if isinstance(c.rhs.numerics, int_numerics.IntNumerics):
            c.rhs.numerics = c.rhs.numerics.replace(round=False)
          # c.lhs.numerics.clip = False
          # c.rhs.numerics.clip = False

        disable_quant(cfg.fwd)
        disable_quant(cfg.dlhs)
        disable_quant(cfg.drhs)
        lhs_quant = not isinstance(cfg.fwd.lhs.numerics, no_numerics.NoNumerics)
        rhs_quant = not isinstance(cfg.fwd.rhs.numerics, no_numerics.NoNumerics)
        if lhs_quant:
          cfg.drhs.rhs.use_fwd_quant = use_fwd_quant
        if rhs_quant:
          cfg.dlhs.rhs.use_fwd_quant = use_fwd_quant
      if local_aqt is not None:
        # Currently we are not supporting local_aqt in fwd pass
        # cfg.fwd.local_aqt = local_aqt
        cfg.dlhs.local_aqt = local_aqt
        cfg.drhs.local_aqt = local_aqt

      # When using abs-max scaling, this should be a no-op.
      if isinstance(cfg.fwd.lhs.numerics, int_numerics.IntNumerics):
        cfg.fwd.lhs.numerics = cfg.fwd.lhs.numerics.replace(
            clip_gradient=clip_gradient
        )
      if isinstance(cfg.fwd.rhs.numerics, int_numerics.IntNumerics):
        cfg.fwd.rhs.numerics = cfg.fwd.rhs.numerics.replace(
            clip_gradient=clip_gradient
        )

      return cfg

    # test dot_general
    lhs = rand_unif(lhs_shape, lhs_maxval, seed, dtype)
    rhs = rand_unif(rhs_shape, rhs_maxval, seed + 1, dtype)
    gra = rand_unif(gra_shape, gra_maxval, seed + 2, dtype)

    def aqt_dg_full(
        use_fake_quant,
        use_fwd_quant=None,
        fwd_lhs_tricky_clip_and_round=False,
        local_aqt=None,
    ):
      cfg = modify_cfg(
          use_fake_quant=use_fake_quant,
          use_fwd_quant=use_fwd_quant,
          fwd_lhs_tricky_clip_and_round=fwd_lhs_tricky_clip_and_round,
          local_aqt=local_aqt,
          clip_gradient=clip_gradient,
      )
      config.set_context(cfg, key=jax.random.PRNGKey(4), train_step=None)
      dg = aqt.make_dot_general(cfg)
      return lambda lhs, rhs: dg(lhs, rhs, dims)

    def aqt_dg_raw(use_fake_quant):
      cfg = modify_cfg(use_fake_quant=use_fake_quant)
      config.set_context(cfg, key=jax.random.PRNGKey(4), train_step=None)
      dg_raw = aqt._make_dot_general_raw(cfg.fwd)
      return lambda lhs, rhs: dg_raw(lhs, rhs, None, None, dims)[0]

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
            lhs_qt,
            rhs_qt,
            dimension_numbers,
        ):
          del lhs_qt, rhs_qt
          if isinstance(rhs, aqt.MultiTensor):
            rhs = rhs.x

          ret = jax.lax.dot_general(lhs, rhs, dimension_numbers)
          ret *= delta

          def res(v):
            return aqt.TensorRes(
                mt=aqt.MultiTensor(
                    x=v,
                    qx=aqt_tensor.QTensor(qvalue=v, scale=None, scale_t=None),
                ),
                quant_grad=None,
            )

          res = aqt.DotGeneralRes(
              lhs=res(lhs),
              rhs=res(rhs),
          )
          return ret, res

        return dg

      m1 = dg_mul(2.0)
      m2 = dg_mul(4.0)
      m3 = dg_mul(8.0)
      out, (out_lhs_qt, out_rhs_qt) = aqt._dot_general_raw_attach_gradient(
          m1, m2, m3
      )(
          lhs,
          rhs,
          None,
          None,
          dims,
      )
      # TODO(lew): Test values instead of del.
      del out_lhs_qt, out_rhs_qt
      return out

    test_jaxpr_dtype(
        lambda: aqt_dg_full(False)(lhs, rhs),
        [modify_cfg().fwd],
        lhs.dtype,
    )
    test_jaxpr_dtype(
        lambda: jax.vjp(aqt_dg_full(False), lhs, rhs),
        [modify_cfg().fwd],
        lhs.dtype,
    )
    _, backprop = jax.vjp(aqt_dg_full(False), lhs, rhs)
    test_jaxpr_dtype(
        lambda: backprop(gra),
        [modify_cfg().dlhs, modify_cfg().drhs],
        gra.dtype,
    )

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
        if "check_fwd_lhs_tricky_clip_and_round" not in options:
          options["check_fwd_lhs_tricky_clip_and_round"] = False

        lrf = dg(lhs, rhs)
        lr, backprop = jax.vjp(dg, lhs, rhs)

        gl, gr = backprop(gra) if options["test_gradient"] else (None, None)

        if options["check_fwd_lhs_tricky_clip_and_round"]:
          # Test that all the expected were zero anyway
          where_zero_gradient_expected = lhs < 0
          assert (gl[where_zero_gradient_expected] == 0.0).all()

        good_lr = lr if good_lr is None else good_lr
        good_gl = gl if good_gl is None else good_gl
        good_gr = gr if good_gr is None else good_gr

        test_eq(f"{name}: lr vs lrf", lr, lrf)

        lr_mult, gl_mult, gr_mult = options["mult"]
        test_eq(f"{name}: lr", good_lr, lr / lr_mult)  # forward pass
        if options["test_gradient"]:
          test_eq(f"{name}: gl", good_gl, gl / gl_mult)  # backward pass  # pytype: disable=unsupported-operands
          test_eq(f"{name}: gr", good_gr, gr / gr_mult)  # backward pass  # pytype: disable=unsupported-operands

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
        (
            "default    ",
            aqt_dg_full(False, local_aqt=config.LocalAqt(2)),
            dict(),
        ),
        (
            "default    ",
            aqt_dg_full(True, local_aqt=config.LocalAqt(2)),
            dict(),
        ),
    ])

    check([
        ("lax_dg    ", lax_dg, dict()),
        ("lax_dg_248", lax_dg_248, dict(mult=(2.0, 4.0, 8.0))),
    ])

    if isinstance(readonly_cfg.fwd.lhs.numerics, int_numerics.IntNumerics):
      check([
          (
              "check_fwd_lhs_tricky_clip_and_round",
              aqt_dg_full(False, fwd_lhs_tricky_clip_and_round=True),
              dict(check_fwd_lhs_tricky_clip_and_round=True),
          ),
      ])

  def test_dynamic_context(self):
    @jax.jit
    def f(lhs, rhs):
      cfg = config.dot_general_make()
      config.set_context(cfg, key=jax.random.PRNGKey(4), train_step=None)
      dg = aqt.make_dot_general(cfg)
      return dg(lhs, rhs, (((0,), (0,)), ((), ())))

    lhs, rhs = jnp.array([3.0, 4.0]), jnp.array([4.0, 5.0])
    jax.value_and_grad(f)(lhs, rhs)

  def test_hardware_int8(self, seed=0):
    cfg = config.dot_general_raw_make(8, 8)

    def dg(lhs, rhs):
      ret, _ = aqt._make_dot_general_raw(cfg)(
          lhs,
          rhs,
          None,
          None,
          (((1,), (0,)), ((), ())),
      )
      return ret

    lhs = rand_unif((10, 20), 1.0, seed)
    rhs = rand_unif((20, 30), 1.0, seed + 1)
    test_jaxpr_dtype(lambda: dg(lhs, rhs), [cfg], lhs.dtype)
    assert cfg.lhs.numerics.get_dtype() == jnp.int8
    assert cfg.rhs.numerics.get_dtype() == jnp.int8

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
    cfg = config.conv_general_dilated_make(2, lhs_bits, rhs_bits)

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
    aqt_conv_fn = aqt_conv.make_conv_general_dilated(cfg)
    kwargs = {
        "window_strides": (1, 1),
        "padding": "SAME",
        "dimension_numbers": fl._conv_dimension_numbers(lhs.shape),
    }
    lhs_fq = aqt_tensor.make_fake_quant(cfg.lhs)(lhs)
    rhs_fq = aqt_tensor.make_fake_quant(cfg.rhs)(rhs)
    prod_fq = lax_conv(lhs_fq, rhs_fq, **kwargs)
    prod_aqt = aqt_conv_fn(lhs, rhs, **kwargs)
    assert (prod_aqt == prod_fq).all()

  @parameterized.parameters([
      dict(
          shard_count=2,
          lhs=[1270.0, 10.0, 1270000.0, 10000.0],
          expected_product=1281280.0,
      ),
      dict(
          shard_count=1,
          lhs=[1270.0, 10.0, 1270000.0, 10000.0],
          expected_product=1280000.0,
      ),
  ])
  def test_local_aqt(self, shard_count, lhs, expected_product):
    # create a config that quantizes both forward and backward passes to int8
    # set the number of shards (local aqt) to 2
    cfg = config.fully_quantized(
        fwd_bits=8,
        bwd_bits=8,
        use_stochastic_rounding=False,
        drhs_local_aqt=config.LocalAqt(shard_count),
    )
    cfg.fwd.lhs.numerics = cfg.fwd.lhs.numerics.replace(preserve_max_val=True)
    cfg.fwd.rhs.numerics = cfg.fwd.rhs.numerics.replace(preserve_max_val=True)
    cfg.drhs.lhs.numerics = cfg.drhs.lhs.numerics.replace(preserve_max_val=True)
    cfg.drhs.rhs.numerics = cfg.drhs.rhs.numerics.replace(preserve_max_val=True)
    dg = lambda lhs, rhs: aqt.make_dot_general(cfg)(
        lhs,
        rhs,
        dimension_numbers=(((), ()), ((), ())),
    )
    lhs = jnp.array(lhs)
    rhs = jnp.array([1.0])
    output, bprop = jax.vjp(dg, lhs, rhs)
    _, drhs = bprop(jnp.ones_like(output))
    assert drhs == expected_product


if __name__ == "__main__":
  absltest.main()
