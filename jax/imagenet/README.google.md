
# How to run

launch ImageNet training on a pod slice:
```
gxm third_party/google_research/google_research/aqt/jax/imagenet/google/xm_launch.py -- --cell=[cell] --platform=[jf/df] --tpu_topology=8x8 --batch_size=8192 -hparams_config_filename=resnet50_bfloat16.py --name="flax_imagenet"
```

If you train on a smaller TPU layout, you need a smaller batch size.

8x8 -> 8192
4x4 -> 4096
2x2 -> 2048

(The general rule is: Scale linearly, but for large
batch sizes with SGD the rule stops working, hence
8x8 -> 8192)

There's no need to change the learning rate -- it's
rescaled linearly in the batch size.


# Using mixed precision training on NVIDIA V100 GPUs
```
gxm third_party/google_research/google_research/aqt/jax/imagenet/google/xm_launch.py -- --platform=gpu \
    --gpu_count=8 --gpu_type=v100 --batch_size=2048 \
    --cell=[cell] --name="imagenet_gpu_f16"
```

### Example launch experiments with XManager

Please refer to `train.py` for all flag options.

#### Dragonfish TPU 8x8

```bash
gxm third_party/google_research/google_research/aqt/jax/imagenet/google/xm_launch.py --cell=tp \
--platform=df --tpu_topology=8x8 --batch_size=8192 \
--hparams_config_filename=resnet50_w8_a8_auto.py --name=imagenet_quant
```

#### Leaderboard

`quant_target` argument used in `hparams_gen.py` | Description                                                     | Example run (on TPUs) ([Comparison](https://tensorboard.corp.google.com/compare/bfloat16:7042384139848754184,weights_only:1589053866757300996,weights_and_fixed_acts:1467147312828514922,weights_and_auto_acts:4947549816151718181/))
------------------------------------------------ | --------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
`NONE`                                           | No quantization; **_default_**                                  | [2020-07-08](https://tensorboard.corp.google.com/experiment/7042384139848754184/)
`WEIGHTS_ONLY`                                   | Weights are quantized only                                      | [2020-08-11](https://tensorboard.corp.google.com/experiment/1589053866757300996/)
`WEIGHTS_AND_FIXED_ACTS`                         | Weights and activations are quantized; no automatic GetBounds   | [2020-08-11](https://tensorboard.corp.google.com/experiment/1467147312828514922/)
`WEIGHTS_AND_AUTO_ACTS`                          | Weights and activations are quantized; with automatic GetBounds | [2020-08-12](https://tensorboard.corp.google.com/experiment/4947549816151718181)
