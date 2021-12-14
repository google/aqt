"""Resnet101 unquantized model."""

from google3.third_party.google_research.google_research.aqt.jax.imagenet.configs import base_config


def get_config(quant_target=base_config.QuantTarget.none):
  return base_config.get_config(
      imagenet_type=base_config.ImagenetType.resnet101,
      quant_target=quant_target)
