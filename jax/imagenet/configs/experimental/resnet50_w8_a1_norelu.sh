#!/bin/bash
gxm ../../google/xm_launch.py \
  --platform=df --tpu_topology=8x8 \
  --batch_size=8192 \
  --hparams_config_filename=experimental/resnet50_w8_a1_norelu.py \
  --name=resnet50-w8a1-baseline \
  --xm_resource_pool=peace --xm_resource_alloc=user:peace/catalyst
