# AQT : Accurate Quantized Training
go/aqt2

AQT is a JAX library has two primary goals:
1. Acceleration of JAX computation using int8, int4 and float8 available on existing hardware.
1. Research on quantization to aid design of future ML hardware.

[Cloud ML AQT introduction](https://cloud.google.com/blog/products/compute/accurate-quantized-training-aqt-for-tpu-v5e/) is a nice well-polished first read.

## AQT value

AQT-int8 allows typically a 20-40% speedup in training time to a solution of
equal quality while outputting a pre-quantized model on platforms with 8-bit
acceleration (VLC and newer).
AQT achieves this by evaluating most of the matmuls in int8 (both fwd pass and bwd pass).

AQT is easy to use with a drop in lax.dot_general replacement and existing integrations into Flax, Pax and MaxText.
AQT was announced at Google Cloud Next, and has existing adoption with Cloud customers.
AQT delivered a key 39.9% speedup in MLPerf training 2023.

AQT training acceleration is applicable to both pre-training and fine-tuning yielding similarly good results.

## Public README.md

Most of the AQT documentation is in the [open-source README.md](README.md).
It contains information:
- list of features,
- links to Cloud ML blog posts and research papers,
- tutorial on how to use it on JAX level,
- the AQT configuration,
- relation of different AQT versions (v1, v2, TF, ...)
- rudimentary description of the internal AQT math.

## How to use AQT in Pax

We don't have a well developed documentation in Pax, but at least we can leave
the reader with some example links:

- This [`cfg_set_quantization(...)`](http://google3/nlp/mum/pax/quantization/experiments.py;l=50;rcl=584723066)
  demonstrates how to inject AQT quantization into one particular model.
- This function [cfg_set_quantization(...)](http://google3/intelligence/mobile_llms/pax/ulm/pretrain.py;l=354;rcl=584722680)
  demonstrates how to use go/fiddle selectors to make it a bit more automatix.

We will try extend a documentation whenever questions appear on AQT Users chat.

## AQT Users chat

Please join [AQT Users](https://chat.google.com/room/AAAASQ4OKpw?cls=1) google internal chat.
Feel free to have any AQT-related discussion there.
Consider starting threads with *bolded title*.

## AQT Catalyst team

AQT is developed by [AQT Catalyst team](https://moma.corp.google.com/team/1448391999960).
