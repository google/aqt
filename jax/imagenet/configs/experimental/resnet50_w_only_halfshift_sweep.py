"""Create a sweep over half_shift flag under different weight precisions.

During the sweep, only weights are quantized.
The weight precisions of conv_init and the final dense layer are set to 8.
"""

import copy
import ml_collections
from google3.third_party.google_research.google_research.aqt.jax.imagenet.configs import base_config


def get_config():
  """Returns sweep configuration (see module docstring)."""
  sweep_config = ml_collections.ConfigDict()
  base_config_dict = base_config.get_config(
      imagenet_type=base_config.ImagenetType.resnet50,
      quant_target=base_config.QuantTarget.weights_only)
  configs = []

  for half_shift in [False, True]:
    for prec in [1, 2, 3, 4]:
      config = copy.deepcopy(base_config_dict)
      config.weight_prec = prec
      config.model_hparams.conv_init.weight_prec = 8
      config.model_hparams.dense_layer.weight_prec = 8
      config.half_shift = half_shift
      config.metadata.hyper_str = f"resnet50_w_only_halfshift_sweep_halfshift_{half_shift}_prec_{prec}"
      configs.append(config)

  sweep_config.configs = configs
  return sweep_config
