#!/bin/bash
gxm ../../../google/xm_launch.py \
  --cell=tp --platform=df --tpu_topology=8x8 \
  --batch_size=8192 \
  --hparams_config_filename=experimental/google/resnet50_w1_a4_auto.py \
  --name=resnet50_w1_a4_auto_baseline \
  --xm_resource_pool=brain --xm_resource_alloc=user:brain/positron-sum
