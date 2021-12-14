#!/bin/bash
gxm ../../../google/xm_launch.py \
  --platform=df --tpu_topology=8x8 \
  --batch_size=8192 \
  --hparams_config_filename=experimental/google/pokebnn_config.py \
  --name=sweep-pokebnn \
  --xm_resource_pool=peace --xm_resource_alloc=user:peace/catalyst
