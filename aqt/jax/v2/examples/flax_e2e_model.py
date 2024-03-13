# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Mnist example."""

import functools
from typing import Any
from absl import app
from aqt.jax.v2 import config as aqt_config
from aqt.jax.v2 import tiled_dot_general
from aqt.jax.v2.flax import aqt_flax
from aqt.jax.v2.flax import utils as aqt_flax_utils
from flax import linen as nn
from flax import struct
from flax.metrics import tensorboard
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_datasets as tfds


class CNN(nn.Module):
  """A simple CNN model."""
  bn_use_stats: bool
  aqt_cfg: aqt_config.DotGeneral
  quant_mode: aqt_flax_utils.QuantMode = aqt_flax_utils.QuantMode.TRAIN

  @nn.compact
  def __call__(self, x):
    tiling_cfg = tiled_dot_general.Cfg(
        lhs=tiled_dot_general.TensorTiling(
            contraction_axes=[
                tiled_dot_general.AxisTiling(
                    axis=1, tile_count=2, tile_size=None
                ),
            ],
            remaining_axes=[],
        ),
        rhs=tiled_dot_general.TensorTiling(
            contraction_axes=[
                tiled_dot_general.AxisTiling(
                    axis=0, tile_count=2, tile_size=None
                ),
            ],
            remaining_axes=[],
        ),
    )
    aqt_dg = functools.partial(
        aqt_flax.AqtDotGeneral,
        self.aqt_cfg,
        # In nn.Dense, it is RHS that has the kernel.
        rhs_quant_mode=self.quant_mode,
        tiling_cfg=tiling_cfg,
    )
    use_running_avg = not self.bn_use_stats
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.BatchNorm(use_running_average=use_running_avg, dtype=x.dtype)(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.BatchNorm(use_running_average=use_running_avg, dtype=x.dtype)(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=256, dot_general_cls=aqt_dg)(x)
    x = nn.relu(x)
    x = nn.Dense(features=10, dot_general_cls=aqt_dg)(x)

    # Simple demonstration of how to quantize einsum.
    identity = jnp.identity(10, dtype=x.dtype)
    einsum = aqt_flax.AqtEinsum(self.aqt_cfg, lhs_quant_mode=self.quant_mode)
    # Note for AQT developers:
    #   This equation is harder because jnp.einsum and einsum swap lhs and rhs.
    x = einsum('bc,ab->ac', identity, x)
    return x


@functools.partial(jax.jit, static_argnums=(3,))
def apply_model(state, images, labels, train):
  """Computes gradients, loss and accuracy for a single batch."""

  cnn = state.cnn_train if train else state.cnn_eval
  def loss_fn(model):
    logits, updated_var = cnn.apply(
        model,
        images,
        rngs={'params': jax.random.PRNGKey(0)},
        mutable=True,
    )
    one_hot = jax.nn.one_hot(labels, 10)
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
    return loss, (logits, updated_var)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True, allow_int=True)
  aux, grads = grad_fn(state.model)
  loss, (logits, updated_var) = aux
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  return grads, loss, accuracy, updated_var


@jax.jit
def update_model(state, grads, updated_var):
  params = state.model['params']
  param_grad = grads['params']
  updates, new_opt_state = state.tx.update(param_grad, state.opt_state, params)
  new_params = optax.apply_updates(params, updates)
  updated_var.update(params=new_params)
  return state.replace(
      model=updated_var,
      opt_state=new_opt_state,
  )


def train_epoch(state, train_ds, batch_size, rng):
  """Train for a single epoch."""
  train_ds_size = len(train_ds['image'])
  steps_per_epoch = train_ds_size // batch_size

  perms = jax.random.permutation(rng, len(train_ds['image']))
  perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
  perms = perms.reshape((steps_per_epoch, batch_size))

  epoch_loss = []
  epoch_accuracy = []

  for perm in perms:
    batch_images = train_ds['image'][perm, ...]
    batch_labels = train_ds['label'][perm, ...]
    grads, loss, accuracy, updated_var = apply_model(
        state, batch_images, batch_labels, train=True
    )
    state = update_model(state, grads, updated_var)
    epoch_loss.append(loss)
    epoch_accuracy.append(accuracy)
  train_loss = np.mean(epoch_loss)
  train_accuracy = np.mean(epoch_accuracy)
  return state, train_loss, train_accuracy


def get_datasets():
  """Load MNIST train and test datasets into memory."""
  print('get_datasets started')
  ds_builder = tfds.builder('mnist')
  ds_builder.download_and_prepare()
  train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
  test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
  train_ds['image'] = jnp.float32(train_ds['image']) / 255.0
  test_ds['image'] = jnp.float32(test_ds['image']) / 255.0
  print('get_datasets DONE')
  return train_ds, test_ds


class TrainState(struct.PyTreeNode):
  """Train state."""

  cnn_train: Any = struct.field(pytree_node=False)
  cnn_eval: Any = struct.field(pytree_node=False)
  model: Any = struct.field(pytree_node=True)
  tx: optax.GradientTransformation = struct.field(pytree_node=False)
  opt_state: optax.OptState = struct.field(pytree_node=True)


def create_train_state(rng, aqt_cfg):
  """Creates initial `TrainState`."""
  cnn_train = CNN(bn_use_stats=True, aqt_cfg=aqt_cfg)
  model = cnn_train.init({'params': rng}, jnp.ones([1, 28, 28, 1]))
  learning_rate = 0.1
  momentum = 0.9
  tx = optax.sgd(learning_rate, momentum)
  cnn_eval = CNN(bn_use_stats=False, aqt_cfg=aqt_cfg)
  return TrainState(
      cnn_train=cnn_train,
      cnn_eval=cnn_eval,
      model=model,
      tx=tx,
      opt_state=tx.init(model['params']),
  )


def train_and_evaluate(
    num_epochs: int, workdir: str, aqt_cfg: aqt_config.DotGeneral
) -> TrainState:
  """Execute model training and evaluation loop."""
  train_ds, test_ds = get_datasets()
  rng = jax.random.key(0)

  summary_writer = tensorboard.SummaryWriter(workdir)

  rng, init_rng = jax.random.split(rng)
  state = create_train_state(init_rng, aqt_cfg)

  batch_size = 128
  for epoch in range(1, num_epochs + 1):
    rng, input_rng = jax.random.split(rng)
    state, train_loss, train_accuracy = train_epoch(
        state, train_ds, batch_size, input_rng
    )
    _, test_loss, test_accuracy, _ = apply_model(
        state, test_ds['image'], test_ds['label'], train=False
    )

    print(
        'epoch:% 3d, train_loss: %.30f, train_accuracy: %.30f, test_loss:'
        ' %.30f, test_accuracy: %.30f'
        % (
            epoch,
            train_loss,
            train_accuracy * 100,
            test_loss,
            test_accuracy * 100,
        ),
        flush=True,
    )

    summary_writer.scalar('train_loss', train_loss, epoch)
    summary_writer.scalar('train_accuracy', train_accuracy, epoch)
    summary_writer.scalar('test_loss', test_loss, epoch)
    summary_writer.scalar('test_accuracy', test_accuracy, epoch)

  summary_writer.flush()
  return state


def serving_conversion(train_state):
  """Model conversion (quantized weights freezing)."""
  aqt_cfg = train_state.cnn_eval.aqt_cfg
  input_shape = (1, 28, 28, 1)
  cnn_freeze = CNN(
      bn_use_stats=False,
      aqt_cfg=aqt_cfg,
      quant_mode=aqt_flax_utils.QuantMode.CONVERT,
  )
  _, model_serving = cnn_freeze.apply(
      train_state.model,
      jnp.ones(input_shape),
      rngs={'params': jax.random.PRNGKey(0)},
      mutable=True,
  )
  cnn_serve = CNN(
      bn_use_stats=False,
      aqt_cfg=aqt_cfg,
      quant_mode=aqt_flax_utils.QuantMode.SERVE,
  )

  return cnn_serve.apply, model_serving


@jax.jit
def serve(state):
  """Take train state, freeze integer weights, and serve."""
  # get sample serving data
  _, test_ds = get_datasets()
  sample_image, sample_label = test_ds['image'][:64], test_ds['label'][:64]
  # serving
  serve_fn, model_serving = serving_conversion(state)
  logits = serve_fn(
      model_serving, sample_image, rngs={'params': jax.random.PRNGKey(0)}
  )
  # compute serving loss
  one_hot = jax.nn.one_hot(sample_label, 10)
  loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
  return loss


def serve_fn_hlo(state):
  """Example on how to inspect HLO to verify if everything was quantized."""
  # get sample serving data
  _, test_ds = get_datasets()
  sample_image = test_ds['image'][:64]
  # serving
  serve_fn, model_serving = serving_conversion(state)
  # The following XLA graph is only needed for debugging purpose
  hlo = jax.xla_computation(serve_fn)(
      model_serving,
      sample_image,
      rngs={'params': jax.random.PRNGKey(0)},
  ).as_hlo_module()
  return hlo


def main(argv):
  del argv
  aqt_cfg = aqt_config.fully_quantized(fwd_bits=8, bwd_bits=8)
  state = train_and_evaluate(
      num_epochs=2, workdir='/tmp/aqt_mnist_example', aqt_cfg=aqt_cfg
  )
  loss = serve(state)
  print('serve loss on sample ds: {}'.format(loss))


if __name__ == '__main__':
  app.run(main)
