# AQT FAQ

[TOC]

## General questions

### Which quantization methods are supported?

AQT supports Quantized Training(QT), Quantization Aware Training(QAT), and
Post-Training Quantization(PTQ). See
[AQT Serving in the main README](https://g3doc.corp.google.com/third_party/py/aqt/README.md#aqt-serving)
to understand how AQT can be used for QAT or PTQ.

### Which ops are supported?

AQT can quantize DotGeneral and Einsum in both the forward and backward passes;
Conv quantization is supported only in the forward pass at the moment.

### Can AQT quantize activation values?

Yes, AQT is currently capable of Dynamic Range Quantization(DRQ). Static Range
Quantization(SRQ) support is coming soon.

### Where can I find examples?

There is
[an e2e Flax example](http://google3/third_party/py/aqt/jax/v2/examples/flax_e2e_model.py)
in the AQT codebase. Checking out usages of
[the DotGeneral config class](http://google3/third_party/py/aqt/jax/v2/aqt_dot_general.py;l=786;rcl=617903381)
is also a good idea.

## Configurations

### Can I configure sub-channel quantization in AQT?

Yes, sub-channel quantization is referred as Local AQT. To enable sub-channel
quantization, you can specify the desired quantization granularity for
contraction axis. Use either the
[`set_local_aqt`](http://google3/third_party/py/aqt/jax/v2/config.py;l=212;rcl=617372892)
configuration function or other config function parameters that end with the
`_local_aqt` suffix.

Another option is using
[`tiled_dot_general`](http://google3/third_party/py/aqt/jax/v2/tiled_dot_general.py;l=234;rcl=617670052).
See references of
[`tiled_dot_general.Cfg`](http://google3/third_party/py/aqt/jax/v2/tiled_dot_general.py;l=57;rcl=618275573)
for examples.

### What `contraction_axis_shard_count` should I use?

It depends on a multitude of factors, but setting it to the number of machines
across which a given contraction axis is sharded on is generally a good starting
point.
