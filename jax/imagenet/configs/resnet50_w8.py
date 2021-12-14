"""Resnet50 weights only quantized to 8 bits model."""

from google3.third_party.google_research.google_research.aqt.jax.imagenet.configs import base_config


def get_config(quant_target=base_config.QuantTarget.weights_only):
  config = base_config.get_config(
      imagenet_type=base_config.ImagenetType.resnet50,
      quant_target=quant_target)
  config.weight_prec = 8
  return config
