"""Tests for imagenet.train."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import random
import jax.numpy as jnp
import numpy as onp

from google3.third_party.google_research.google_research.aqt.jax.imagenet import hparams_config
from google3.third_party.google_research.google_research.aqt.jax.imagenet import models
from google3.third_party.google_research.google_research.aqt.jax.imagenet import train_utils
from google3.third_party.google_research.google_research.aqt.jax.imagenet.configs import resnet50_w4
from google3.third_party.google_research.google_research.aqt.jax.imagenet.configs import resnet50_w4_a4_fixed
from google3.third_party.google_research.google_research.aqt.jax.imagenet.configs.paper import resnet50_bfloat16
from google3.third_party.google_research.google_research.aqt.jax.imagenet.configs.paper import resnet50_w4_a4_auto
from google3.third_party.google_research.google_research.aqt.utils import hparams_utils


class TrainTest(parameterized.TestCase):
  @parameterized.named_parameters(
      dict(
          testcase_name='quantization_none',
          base_config_filename=resnet50_bfloat16),
      dict(
          testcase_name='quantization_weights_only',
          base_config_filename=resnet50_w4),
      dict(
          testcase_name='quantization_weights_and_fixed_acts',
          base_config_filename=resnet50_w4_a4_fixed),
      dict(
          testcase_name='quantization_weights_and_auto_acts',
          base_config_filename=resnet50_w4_a4_auto),
  )  # pylint: disable=line-too-long

  def test_create_model(self, base_config_filename):
    hparams = hparams_utils.load_hparams_from_config_dict(
        hparams_config.TrainingHParams, models.ResNet.HParams,
        base_config_filename.get_config())
    model, state = train_utils.create_model(
        random.PRNGKey(0),
        8,
        224,
        jnp.float32,
        hparams.model_hparams,
        train=True)
    x = random.normal(random.PRNGKey(1), (8, 224, 224, 3))
    y, new_state = model.apply(state, x, mutable=True)
    state = jax.tree_map(onp.shape, state)
    new_state = jax.tree_map(onp.shape, new_state)
    self.assertEqual(state, new_state)
    self.assertEqual(y.shape, (8, 1000))


if __name__ == '__main__':
  absltest.main()
