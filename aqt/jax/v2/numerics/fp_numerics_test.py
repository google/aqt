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

from absl.testing import absltest
from absl.testing import parameterized
from aqt.jax.v2 import utils
from aqt.jax.v2.numerics import fp_numerics
import jax
import jax.numpy as jnp

jax.config.update("jax_numpy_rank_promotion", "raise")


def assert_equal(x, expected_x, error_msg="", aux_x=None, mask=None):
  x = jnp.array(x)
  expected_x = jnp.array(expected_x)
  sh = x.shape
  assert expected_x.shape == sh, f"{sh=}, {expected_x.shape=}"
  if mask is not None:
    assert mask.shape == sh, f"{sh=}, {mask.shape=}"
    x = jnp.where(mask, x, jnp.zeros_like(x))
    expected_x = jnp.where(mask, expected_x, jnp.zeros_like(expected_x))
  if aux_x is not None:
    assert aux_x.shape == sh, f"{sh=}, {aux_x.shape=}"
  if len(sh) > 1:
    assert len(sh) == 2
    x = x.reshape((sh[0] * sh[1],))
    expected_x = expected_x.reshape((sh[0] * sh[1],))
    if aux_x is not None:
      aux_x = aux_x.reshape((sh[0] * sh[1],))
  (all_count,) = x.shape

  correct_val = x == expected_x
  correct_nan = jnp.isnan(x) & jnp.isnan(expected_x)
  correct = correct_val | correct_nan
  correct_count = jnp.sum(correct)
  bad = ~correct
  index_bad = (jnp.arange(all_count))[bad]
  values_bad = x[bad]
  expected_x_bad = expected_x[bad]
  error = (
      error_msg
      + "\n"
      + f"correct_count     = {correct_count}/{all_count}\n"
      f"!eq index         = {index_bad}\n"
      f"!eq lhs(actual)   = {values_bad}\n"
      f"!eq rhs(expected) = {expected_x_bad}\n"
  )
  if aux_x is not None:
    aux_x_bad = aux_x[bad]
    error += f"aux_x_bad          = {aux_x_bad}\n"
  assert correct_count == all_count, error


def fp_values(cfg: fp_numerics.FpNumericsConfig):
  nexp = cfg.nexp
  minexp = cfg.minexp
  nmant = cfg.nmant
  has_subnormals = cfg.has_subnormals
  has_two_nan = cfg.has_two_nan
  has_naninf = cfg.has_naninf

  assert not (has_two_nan and has_naninf)
  bits = 1 + nexp + nmant

  bitmasks = jnp.arange(2**bits, dtype=jnp.uint16)
  sign_bits = bitmasks >> (nexp + nmant)
  exp_bits = (bitmasks >> nmant) & ((1 << nexp) - 1)
  man_bits = bitmasks & ((1 << nmant) - 1)

  sign = sign_bits.astype(jnp.float32) * -2.0 + 1
  exp = exp_bits.astype(jnp.float32)
  man = man_bits.astype(jnp.float32)

  if cfg == fp_numerics.RADIX4:
    values = 4 ** jnp.array([-3, -2, -1, 0, 1, 2, 3], dtype=jnp.bfloat16)
    values = jnp.concatenate([jnp.zeros(1), values, -jnp.zeros(1), -values])
    return bitmasks, values

  if has_subnormals:
    exp -= 1  # resereve exp=0

  values = sign * 2 ** (exp + minexp) * (1 + man / 2**nmant)
  if has_subnormals:
    subnormals = sign * 2**minexp * (man / 2**nmant)
    values = jnp.where(exp_bits == 0, subnormals, values)
  if has_two_nan:
    is_nan = (exp_bits == 2**nexp - 1) & (man == 2**nmant - 1)
    values = jnp.where(is_nan, jnp.nan, values)
  if has_naninf:
    is_nan = exp_bits == 2**nexp - 1
    values = jnp.where(is_nan, jnp.nan, values)
    is_inf = (exp_bits == 2**nexp - 1) & (man == 0)
    values = jnp.where(is_inf, sign * jnp.inf, values)

  return bitmasks, values


class FpTest(parameterized.TestCase):

  @parameterized.parameters([
      dict(
          dtype=jnp.float8_e4m3fn,
          cfg=fp_numerics.float8_e4m3fn,
      ),
      dict(
          dtype=jnp.float8_e5m2,
          cfg=fp_numerics.float8_e5m2,
      ),
      dict(
          dtype=jnp.float16,
          cfg=fp_numerics.float16,
      ),
  ])
  def test_fp_values_on_extisting_dtypes(
      self,
      *,
      dtype,
      cfg,
  ):
    def fp_values_dtype(dtype):
      bits = jnp.finfo(dtype).bits
      match bits:
        case 8:
          mask_dtype = jnp.uint8
        case 16:
          mask_dtype = jnp.uint16
        case 32:
          mask_dtype = jnp.uint32
        case 64:
          mask_dtype = jnp.uint64
        case _:
          raise ValueError(f"Unsupported dtype: {dtype}")
      bitmasks = jnp.arange(2**bits, dtype=mask_dtype)
      values = jax.lax.bitcast_convert_type(bitmasks, dtype)
      values = values.astype(jnp.float32)
      return bitmasks, values

    nexp = jnp.finfo(dtype).nexp
    minexp = jnp.finfo(dtype).minexp
    nmant = jnp.finfo(dtype).nmant
    assert nexp == cfg.nexp
    assert minexp == cfg.minexp
    assert nmant == cfg.nmant
    bitmasks, values = fp_values(cfg)
    bitmasks1, expected_values = fp_values_dtype(dtype)
    assert (bitmasks == bitmasks1).all()
    assert_equal(
        values,
        expected_values,
        f" {cfg=}",
    )

  @parameterized.parameters([
      dict(
          nexp=3, minexp=0, nmant=0, has_subnormals=False,
          expected_values=[
              1, 2, 4, 8, 16, 32, 64, 128,
              -1, -2, -4, -8, -16, -32, -64, -128,
          ],
      ),
      dict(
          nexp=3, minexp=0, nmant=0, has_subnormals=True,
          expected_values=[
              0, 1, 2, 4, 8, 16, 32, 64,
              -0, -1, -2, -4, -8, -16, -32, -64,
          ],
      ),
      dict(
          nexp=3, minexp=-2, nmant=0, has_subnormals=False,
          expected_values=[
              0.25, 0.5, 1, 2, 4, 8, 16, 32,
              -0.25, -0.5, -1, -2, -4, -8, -16, -32,
          ],
      ),
      dict(
          nexp=3, minexp=-2, nmant=0, has_subnormals=True,
          expected_values=[
              0, 0.25, 0.5, 1, 2, 4, 8, 16,
              -0, -0.25, -0.5, -1, -2, -4, -8, -16,
          ],
      ),
      dict(
          nexp=2, minexp=0, nmant=0, has_subnormals=False,
          expected_values=[
              1, 2, 4, 8,
              -1, -2, -4, -8,
          ],
      ),
      dict(
          nexp=2, minexp=0, nmant=1, has_subnormals=False,
          expected_values=[
              1, 1.5, 2, 3, 4, 6, 8, 12,
              -1, -1.5, -2, -3, -4, -6, -8, -12,
          ],
      ),
      dict(
          nexp=2, minexp=0, nmant=1, has_subnormals=True,
          expected_values=[
              0, 0.5, 1, 1.5, 2, 3, 4, 6,
              -0, -0.5, -1, -1.5, -2, -3, -4, -6,
          ],
      ),
      dict(
          nexp=1, minexp=0, nmant=2, has_subnormals=True,
          expected_values=[
              0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75,
              -0, -0.25, -0.5, -0.75, -1, -1.25, -1.5, -1.75,
          ],
      ),
      dict(
          nexp=0, minexp=0, nmant=3, has_subnormals=True,
          expected_values=[
              0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875,
              -0, -0.125, -0.25, -0.375, -0.5, -0.625, -0.75, -0.875,
          ],
      ),
      dict(
          nexp=3, minexp=0, nmant=0, has_subnormals=True,
          expected_values=[
              0, 4**-3, 4**-2, 4**-1, 4**0, 4**1, 4**2, 4**3,
              -0, -4**-3, -4**-2, -4**-1, -4**0, -4**1, -4**2, -4**3,
          ],
          radix=4,
      ),
  ])  # pyformat: disable
  def test_fp_some_fp_values(
      self,
      nexp,
      minexp,
      nmant,
      has_subnormals,
      expected_values,
      radix=2,
  ):
    cfg = fp_numerics.FpNumericsConfig(
        nexp=nexp,
        minexp=minexp,
        nmant=nmant,
        has_subnormals=has_subnormals,
        has_two_nan=False,
        has_naninf=False,
        radix=radix,
    )

    bitmasks, values = fp_values(cfg)
    bits = 1 + nexp + nmant
    assert (bitmasks == jnp.arange(2**bits)).all()
    assert_equal(values, expected_values, f"{nexp=}, {minexp=}, {nmant=}")

  # For now we just test:
  #   has_two_nan=False,
  #   has_naninf=False,
  @parameterized.parameters([
      # FP4
      dict(nexp=3, minexp=0, nmant=0, has_subnormals=False),
      dict(nexp=3, minexp=0, nmant=0, has_subnormals=True),
      # dict(nexp=3, minexp=-2, nmant=0, has_subnormals=False),
      # dict(nexp=3, minexp=-2, nmant=0, has_subnormals=True),
      dict(nexp=2, minexp=0, nmant=1, has_subnormals=False),
      dict(nexp=2, minexp=0, nmant=1, has_subnormals=True),
      dict(nexp=1, minexp=0, nmant=2, has_subnormals=False),
      dict(nexp=1, minexp=0, nmant=2, has_subnormals=True),
      dict(nexp=0, minexp=0, nmant=3, has_subnormals=False),
      dict(nexp=0, minexp=0, nmant=3, has_subnormals=True),
      # FP6
      dict(nexp=5, minexp=0, nmant=0, has_subnormals=True),
      dict(nexp=5, minexp=0, nmant=0, has_subnormals=False),
      dict(nexp=4, minexp=0, nmant=1, has_subnormals=True),
      dict(nexp=4, minexp=0, nmant=1, has_subnormals=False),
      dict(nexp=3, minexp=0, nmant=2, has_subnormals=True),
      dict(nexp=3, minexp=0, nmant=2, has_subnormals=False),
      dict(nexp=2, minexp=0, nmant=3, has_subnormals=True),
      dict(nexp=2, minexp=0, nmant=3, has_subnormals=False),
      dict(nexp=1, minexp=0, nmant=4, has_subnormals=True),
      dict(nexp=1, minexp=0, nmant=4, has_subnormals=False),
      dict(nexp=0, minexp=0, nmant=5, has_subnormals=True),
      dict(nexp=0, minexp=0, nmant=5, has_subnormals=False),
      # FP5
      dict(nexp=4, minexp=0, nmant=0, has_subnormals=True),
      dict(nexp=4, minexp=0, nmant=0, has_subnormals=False),
      dict(nexp=3, minexp=0, nmant=1, has_subnormals=True),
      dict(nexp=3, minexp=0, nmant=1, has_subnormals=False),
      dict(nexp=2, minexp=0, nmant=2, has_subnormals=True),
      dict(nexp=2, minexp=0, nmant=2, has_subnormals=False),
      dict(nexp=1, minexp=0, nmant=3, has_subnormals=True),
      dict(nexp=1, minexp=0, nmant=3, has_subnormals=False),
      dict(nexp=0, minexp=0, nmant=4, has_subnormals=True),
      dict(nexp=0, minexp=0, nmant=4, has_subnormals=False),
      # Misc
      dict(nexp=2, minexp=0, nmant=0, has_subnormals=False),
      dict(nexp=3, minexp=0, nmant=0, has_subnormals=True, radix=4),
  ])
  def test_fp_round(
      self,
      nexp,
      minexp,
      nmant,
      has_subnormals,
      radix=2,
      det_x_count=2**7,  # needs to be power of 2, test_noise_axis exact eq
      x_count=128,  # TODO(lew): Test fails for 100, why?
      sr_sample_count=1000000,
      # TODO(lew): Enable these:
      # x_count=2**12,  # needs to be power of 2
      # sr_sample_count=1000,
  ):
    det_x_count += 1  # endpoint
    x_count += 1  # endpoint
    cfg = fp_numerics.FpNumericsConfig(
        nexp=nexp,
        minexp=minexp,
        nmant=nmant,
        has_subnormals=has_subnormals,
        has_two_nan=False,
        has_naninf=False,
        radix=radix,
    )
    _, bucket_centers = fp_values(cfg)
    assert bucket_centers.dtype == jnp.float32
    # bucket_centers = bucket_centers.astype(jnp.float32)
    bucket_centers = bucket_centers.sort()
    assert bucket_centers[-1] == fp_numerics.fp_largest_representable(cfg), (
        bucket_centers[-1],
        fp_numerics.fp_largest_representable(cfg),
    )
    bucket_boundaries = (bucket_centers[1:] + bucket_centers[:-1]) / 2
    assert (
        -bucket_boundaries[0] == bucket_boundaries[-1]
    ), f"Symmetric datatype {bucket_boundaries[0]} vs {bucket_boundaries[-1]}"
    last_bucket_half_size = bucket_centers[-1] - bucket_boundaries[-1]
    assert last_bucket_half_size > 0
    edge_of_last_bucket = jnp.array(
        [bucket_centers[-1] + last_bucket_half_size]
    )
    bucket_boundaries = jnp.concatenate(
        [-edge_of_last_bucket, bucket_boundaries, edge_of_last_bucket]
    )

    # TEST: Check that each interval is rounded to the center.

    bucket_lower = bucket_boundaries[:-1]
    bucket_upper = bucket_boundaries[1:]
    if radix == 4:
      # For 4-bit radix4, the edge between bucket 4**-3 and bucket 0 is not
      # (4**-3 + 0) / 2. The edge is (4**-3 + 4**-4) / 2.
      # That is where floor(log4(1.6 * x)) isexactly -3.
      bucket_lower = jnp.where(
          jnp.abs(bucket_lower) == (4**-3 / 2),
          jnp.sign(bucket_lower) * (4**-3 + 4**-4) / 2,
          bucket_lower,
      )
      bucket_upper = jnp.where(
          jnp.abs(bucket_upper) == (4**-3 / 2),
          jnp.sign(bucket_upper) * (4**-3 + 4**-4) / 2,
          bucket_upper,
      )
    data = jnp.linspace(
        bucket_lower,
        bucket_upper,
        x_count,
        endpoint=True,
        axis=0,
        dtype=jnp.float32,
    )
    assert data.shape == (x_count, 2 ** (1 + nexp + nmant))
    # TODO(lew): Remove when fp_round supports float32. Also below.
    data = data.astype(jnp.bfloat16)
    bucked_interior_mask = jnp.logical_and(
        data != bucket_lower[jnp.newaxis, :],
        data != bucket_upper[jnp.newaxis, :],
    )
    qdata = fp_numerics.fp_round(
        data,
        cfg=cfg,
        key=None,
        stochastic_rounding=False,
    )
    assert qdata.shape == data.shape, f"{qdata.shape=}, {data.shape=}"
    bucket_centers_1 = jnp.expand_dims(bucket_centers, axis=0)
    bucket_centers_1 = jnp.broadcast_to(bucket_centers_1, qdata.shape)
    assert_equal(
        qdata,
        bucket_centers_1,
        f"Config: {nexp=}, {minexp=}, {nmant=}, {has_subnormals=}\n"
        + f"{bucket_centers=}\n"
        + f"{data=}\n"
        + f"{qdata=}\n\n",
        aux_x=data,
        mask=bucked_interior_mask,
    )

    # TEST: Check that overflows are clamped to edge_of_last_bucket.
    for sign in [-1, 1]:
      data = jnp.linspace(
          sign * edge_of_last_bucket,
          sign * 10 * edge_of_last_bucket,
          x_count,
          endpoint=True,
          dtype=jnp.float32,
      )
      data = data.astype(jnp.bfloat16)
      qdata = fp_numerics.fp_round(
          data,
          cfg=cfg,
          key=None,
          stochastic_rounding=False,
      )
      last_center = sign * bucket_centers[-1]

      assert_equal(
          qdata,
          jnp.full_like(qdata, last_center),
          f"Config: {nexp=}, {minexp=}, {nmant=}, {has_subnormals=}\n"
          + f"{bucket_centers=}\n"
          + f"{data=}\n"
          + f"{qdata=}\n\n",
          aux_x=data,
      )

    # TEST: Equi-spaced noise deterministic "SR"

    x = jnp.linspace(
        bucket_centers[0],
        bucket_centers[-1],
        det_x_count,
        endpoint=True,
        dtype=jnp.float32,
    ).astype(jnp.bfloat16)
    # The number of broadcasts should be consistent with the effective noise bit
    # For example, if using 16-bit noise, inputs should be replicated by 2**16.
    bx = jnp.broadcast_to(x[:, jnp.newaxis], (det_x_count, 2**16))
    deterministic_qx = fp_numerics.fp_round(
        bx.astype(jnp.bfloat16),
        cfg=cfg,
        key=None,
        test_noise_axis=1,
        stochastic_rounding=True,
    )
    mean_det_qx = jnp.mean(deterministic_qx, axis=1)
    assert_equal(mean_det_qx, x, f"{cfg=}")

    # TEST: Unbiased stochastic rounding

    x = jnp.linspace(
        bucket_centers[0],
        bucket_centers[-1],
        x_count,
        endpoint=True,
        dtype=jnp.float32,
    ).astype(jnp.bfloat16)
    bx = jnp.broadcast_to(x[:, jnp.newaxis], (x_count, sr_sample_count))
    qx = fp_numerics.fp_round(
        bx.astype(jnp.bfloat16),
        cfg=cfg,
        key=jax.random.PRNGKey(42),
        stochastic_rounding=True,
    )

    assert qx.dtype == jnp.bfloat16
    qx = qx.astype(jnp.float32)
    # qx can hit at most 2 values, larger and smaller bucket center
    qx_smaller = jnp.min(qx, axis=1, keepdims=True)
    qx_larger = jnp.max(qx, axis=1, keepdims=True)
    assert ((qx == qx_larger) | (qx == qx_smaller)).all()

    # Compute empirical probability of rounding up and down.
    qx_smaller_count = jnp.sum(qx == qx_smaller, axis=1, keepdims=True)
    qx_larger_count = jnp.sum(qx == qx_larger, axis=1, keepdims=True)
    total_count = qx_smaller_count + qx_larger_count
    assert_equal(qx_smaller != qx_larger, total_count == sr_sample_count, "zz")

    total_count = jnp.where(  # if we hit bucket center exactly.
        qx_smaller == qx_larger, total_count / 2, total_count
    )
    assert_equal(
        total_count,
        jnp.full_like(total_count, sr_sample_count),
        "",
    )
    p_smaller = qx_smaller_count / sr_sample_count
    bucket_size = qx_larger - qx_smaller
    expected_p_smaller = (qx_larger - x[:, jnp.newaxis]) / bucket_size
    expected_p_smaller = jnp.where(bucket_size == 0, 1.0, expected_p_smaller)
    p_err = jnp.abs(p_smaller - expected_p_smaller)
    p_stderr = p_err * jnp.sqrt(sr_sample_count)
    # print(f"{p_stderr=}")
    assert (p_stderr < 2.0).all(), p_stderr

  @parameterized.parameters([
      # FP4
      dict(cfg=fp_numerics.e3m0, sr=False),
      dict(cfg=fp_numerics.e2m1, sr=False),
      dict(cfg=fp_numerics.e1m2, sr=False),
      dict(cfg=fp_numerics.e0m3, sr=False),
      dict(cfg=fp_numerics.e3m0_ocp, sr=False),
      dict(cfg=fp_numerics.e2m1_ocp, sr=False),
      dict(cfg=fp_numerics.e1m2_ocp, sr=False),
      dict(cfg=fp_numerics.e0m3_ocp, sr=False),
      dict(cfg=fp_numerics.e3m0, sr=True),
      dict(cfg=fp_numerics.e2m1, sr=True),
      dict(cfg=fp_numerics.e1m2, sr=True),
      dict(cfg=fp_numerics.e0m3, sr=True),
      dict(cfg=fp_numerics.e3m0_ocp, sr=True),
      dict(cfg=fp_numerics.e2m1_ocp, sr=True),
      dict(cfg=fp_numerics.e1m2_ocp, sr=True),
      dict(cfg=fp_numerics.e0m3_ocp, sr=True),
      # FP5
      dict(cfg=fp_numerics.e4m0_ocp, sr=False),
      dict(cfg=fp_numerics.e3m1_ocp, sr=False),
      dict(cfg=fp_numerics.e2m2_ocp, sr=False),
      dict(cfg=fp_numerics.e1m3_ocp, sr=False),
      dict(cfg=fp_numerics.e0m4_ocp, sr=False),
      dict(cfg=fp_numerics.e4m0_ocp, sr=True),
      dict(cfg=fp_numerics.e3m1_ocp, sr=True),
      dict(cfg=fp_numerics.e2m2_ocp, sr=True),
      dict(cfg=fp_numerics.e1m3_ocp, sr=True),
      dict(cfg=fp_numerics.e0m4_ocp, sr=True),
      # FP6
      dict(cfg=fp_numerics.e5m0_ocp, sr=False),
      dict(cfg=fp_numerics.e4m1_ocp, sr=False),
      dict(cfg=fp_numerics.e3m2_ocp, sr=False),
      dict(cfg=fp_numerics.e2m3_ocp, sr=False),
      dict(cfg=fp_numerics.e1m4_ocp, sr=False),
      dict(cfg=fp_numerics.e0m5_ocp, sr=False),
      dict(cfg=fp_numerics.e5m0_ocp, sr=True),
      dict(cfg=fp_numerics.e4m1_ocp, sr=True),
      dict(cfg=fp_numerics.e3m2_ocp, sr=True),
      dict(cfg=fp_numerics.e2m3_ocp, sr=True),
      dict(cfg=fp_numerics.e1m4_ocp, sr=True),
      dict(cfg=fp_numerics.e0m5_ocp, sr=True),
      # minexp not implemented, hence the following cfgs are skipped
      # dict(cfg=fp_numerics.float8_e4m3fn),
      # dict(cfg=fp_numerics.float8_e5m2),
      # dict(cfg=fp_numerics.float16),
      # Misc
      dict(cfg=fp_numerics.e1m0, sr=False),
      dict(cfg=fp_numerics.e1m0, sr=True),
      dict(cfg=fp_numerics.e1m2, sr=False),
      dict(cfg=fp_numerics.e1m2, sr=True),
  ])
  def test_fp_round_old_vs_new(
      self, cfg: fp_numerics.FpNumericsConfig, sr: bool
  ):
    print(cfg)
    print(f"stochastic_rounding: {sr}")
    x = jnp.arange(0, 2**16, step=1, dtype=jnp.uint16)
    x = jax.lax.bitcast_convert_type(x, jnp.bfloat16)
    nan_mask = jnp.isnan(x)
    if sr:
      for key_num in [0, 12, 42, 17, 96, 101]:
        key = jax.random.PRNGKey(key_num)
        out = fp_numerics.fp_round(x, cfg=cfg, key=key, stochastic_rounding=sr)
        out_new = fp_numerics.fp_round_new(
            x, cfg=cfg, key=key, stochastic_rounding=sr
        )
        assert (out[~nan_mask] == out_new[~nan_mask]).all()
        assert jnp.isnan(out_new[nan_mask]).all()
    else:
      out = fp_numerics.fp_round(x, cfg=cfg, key=None, stochastic_rounding=sr)
      out_new = fp_numerics.fp_round_new(
          x, cfg=cfg, key=None, stochastic_rounding=sr
      )
      assert (out[~nan_mask] == out_new[~nan_mask]).all()
      assert jnp.isnan(out_new[nan_mask]).all()

  def test_e1m2_vs_e0m3(self):
    e1m2 = fp_numerics.FpNumerics(cfg=fp_numerics.e1m2_ocp)
    e0m3 = fp_numerics.FpNumerics(cfg=fp_numerics.e0m3_ocp)
    context = utils.Context(key=None, train_step=None)
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, shape=(64, 64), dtype=jnp.bfloat16)
    x = x / jnp.abs(x).max()
    e1m2_input = x * e1m2.get_quant_bound()
    e0m3_input = x * e0m3.get_quant_bound()
    assert (e1m2_input == (e0m3_input * 2)).all()
    y_e1m2, _ = e1m2.vjp_fwd(e1m2_input, context=context)
    y_e0m3, _ = e0m3.vjp_fwd(e0m3_input, context=context)
    assert (y_e1m2 == y_e0m3 * 2).all()

if __name__ == "__main__":
  absltest.main()
