"""Cost tests for imagenet.models."""

import logging

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from jax import random
import jax.numpy as jnp

from google3.third_party.google_research.google_research.aqt.jax import compute_cost_utils
from google3.third_party.google_research.google_research.aqt.jax import flax_layers as aqt_flax_layers  # pylint: disable=unused-import
from google3.third_party.google_research.google_research.aqt.jax import hlo_utils
from google3.third_party.google_research.google_research.aqt.jax.imagenet import hparams_config
from google3.third_party.google_research.google_research.aqt.jax.imagenet import models
from google3.third_party.google_research.google_research.aqt.jax.imagenet.configs import resnet50_w4
from google3.third_party.google_research.google_research.aqt.jax.imagenet.configs import resnet50_w4_a2_fixed
from google3.third_party.google_research.google_research.aqt.jax.imagenet.configs import resnet50_w4_a4_fixed
from google3.third_party.google_research.google_research.aqt.jax.imagenet.configs import resnet50_w8
from google3.third_party.google_research.google_research.aqt.jax.imagenet.configs import resnet50_w8_a8_fixed
from google3.third_party.google_research.google_research.aqt.jax.imagenet.configs.paper import resnet50_bfloat16
from google3.third_party.google_research.google_research.aqt.jax.imagenet.train_utils import create_model
from google3.third_party.google_research.google_research.aqt.utils import hparams_utils

FLAGS = flags.FLAGS


class CostTest(parameterized.TestCase):

  def setUp(self):
    super(CostTest, self).setUp()
    self.rng_key = random.PRNGKey(0)

  def _create_hlo_from_resnet_hparams(self, hparams, input_shape):
    """Create an HLO representation from ResNet model and input_shape."""

    # Create model
    rng = random.PRNGKey(0)
    model, init_state = create_model(
        rng,
        input_shape[0],
        input_shape[1],
        jnp.float32,
        hparams.model_hparams,
        train=False)

    # Create HLO
    hlo_proto = hlo_utils.load_hlo_proto_from_model(model, init_state,
                                                    [input_shape])

    del model, init_state
    return hlo_proto

  @parameterized.named_parameters(
      # Unquantized
      dict(
          testcase_name='resnet50_no_quantization',
          base_config_filename=resnet50_bfloat16,
          expected_compute_cost=1046831169536,
          expected_compute_cost_ratio=1.0,
          expected_memory_cost=408046592,
          expected_memory_cost_ratio=1.0,
      ),
      # Weights only
      dict(
          testcase_name='resnet50_full_quantization_weights_only_8',
          base_config_filename=resnet50_w8,
          expected_compute_cost=523415584768,
          expected_compute_cost_ratio=0.5,
          expected_memory_cost=204023296,
          expected_memory_cost_ratio=0.5,
      ),
      dict(
          testcase_name='resnet50_full_quantization_weights_only_4',
          base_config_filename=resnet50_w4,
          expected_compute_cost=261707792384,
          expected_compute_cost_ratio=0.25,
          expected_memory_cost=102011648,
          expected_memory_cost_ratio=0.25,
      ),
      # Weights and activations (fixed)
      dict(
          testcase_name='resnet50_full_quantization_weights_8_fixed_acts_8',
          base_config_filename=resnet50_w8_a8_fixed,
          expected_compute_cost=261707792384,
          expected_compute_cost_ratio=0.25,
          expected_memory_cost=204023296,
          expected_memory_cost_ratio=0.5,
      ),
      dict(
          testcase_name='resnet50_full_quantization_weights_4_fixed_acts_4',
          base_config_filename=resnet50_w4_a4_fixed,
          expected_compute_cost=65426948096,
          expected_compute_cost_ratio=0.0625,
          expected_memory_cost=102011648,
          expected_memory_cost_ratio=0.25,
      ),
      dict(
          testcase_name='resnet50_full_quantization_weights_4_fixed_acts_2',
          base_config_filename=resnet50_w4_a2_fixed,
          expected_compute_cost=32713474048,
          expected_compute_cost_ratio=0.03125,
          expected_memory_cost=102011648,
          expected_memory_cost_ratio=0.25,
      ),
  )  # pylint: disable=line-too-long
  def test_estimate_resnet_cost(self, base_config_filename,
                                expected_compute_cost,
                                expected_compute_cost_ratio,
                                expected_memory_cost,
                                expected_memory_cost_ratio):
    batch_size = 1024
    image_size = 224
    input_channels = 3
    input_shape = (batch_size, image_size, image_size, input_channels)

    logging.info('Testing for %s...', base_config_filename)
    hparams = hparams_utils.load_hparams_from_config_dict(
        hparams_config.TrainingHParams, models.ResNet.HParams,
        base_config_filename.get_config())

    hlo_proto = self._create_hlo_from_resnet_hparams(hparams, input_shape)
    compute_result = compute_cost_utils.estimate_compute_cost(hlo_proto)
    memory_result = compute_cost_utils.estimate_memory_cost(hlo_proto)
    self.assertEqual(compute_result['compute_cost'], expected_compute_cost)
    self.assertAlmostEqual(compute_result['compute_cost_ratio_to_bfloat16'],
                           expected_compute_cost_ratio)
    self.assertEqual(memory_result['memory_cost'], expected_memory_cost)
    self.assertAlmostEqual(memory_result['memory_cost_ratio_to_bfloat16'],
                           expected_memory_cost_ratio)


if __name__ == '__main__':
  FLAGS.metadata_enabled = True  # Passes quantization information to HLO
  absltest.main()
