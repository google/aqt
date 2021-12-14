# Copyright 2021 Google LLC
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

"""XManager script that launches a ImageNet a TPU or GPU job on Borg."""

import collections
import datetime
import functools
import getpass
import operator

from absl import app
from absl import flags

from google3.learning.deepmind.python.adhoc_import import binary_import
# pylint: disable=g-import-not-at-top
try:
  import google3.learning.deepmind.xmanager2.client.google as xm  # pytype: disable=module-attr
except ImportError:
  import google3.learning.deepmind.xmanager as xm

with binary_import.AutoGoogle3():
  from google3.googlex.positron.tensorflow.jax.wmt_mlperf import xm_launch_lib
  from google3.learning.brain.frameworks.xmanager import xm_helper
  from google3.pyglib.contrib.gpathlib import gpath
  from google3.pyglib.contrib.gpathlib import gpath_flag

# jf and df are just short aliases for jellyfish and dragonfish and are
# completely interchangeable
PLATFORMS = ['jf', 'jellyfish', 'df', 'dragonfish', 'gpu']

FLAGS = flags.FLAGS

flags.DEFINE_string('name', 'imagenet_example', 'experiment name')
flags.DEFINE_string('cell', None, 'Tpu cell. If None, will autoselect cell.')
flags.DEFINE_enum('platform', PLATFORMS[0], PLATFORMS, 'Accelerator type')
flags.DEFINE_integer('gpu_count', 1, 'Number of GPUs (if platform=="gpu").')
flags.DEFINE_enum('gpu_type', None, xm.GpuType.values,
                  'GPU type (if platform=="gpu").')
flags.DEFINE_bool('half_precision', True, 'Use 16bit floating point types.')
flags.DEFINE_string(
    'hparams_config_filename',
    default=None,
    help='Path to a python file specifying a TrainingHparam.'
    'If relative, it is interpreted as relative to hparams_config_dir.')

gpath_flag.DEFINE_path(
    'hparams_config_dir',
    'third_party/google_research/google_research/aqt/jax/imagenet/configs/',
    'Path to folder contains hparams config files specified '
    'via hparams_config_filename.')

flags.DEFINE_string(
    'report_dir',
    default='/cns/tp-d/home/cerebra-catalyst/exp_reports/imagenet',
    help=('Directory to save experiment report to after training has '
          'completed. '))

flags.DEFINE_string(
    'resnet508b_ckpt_path',
    default='/cns/tp-d/home/yichizh/flax_imagenet/resnet50-w8a1-baseline_20210727_202401/2-config_idx=1,hparams_config_dict=third_party-google_research-google_research-aqt-jax-imagenet-configs-experimental-resnet50_w8_a1_norelu.py/',
    help=('Path to the checkpoint directory of ResNet508b as a teacher model.'))

flags.DEFINE_integer('batch_size', None, 'batch size"')
flags.mark_flag_as_required('batch_size')

if FLAGS.cell is None:
  FLAGS.cell = xm.Borg.default_cell_selector()

_TPU_BORG_CONFIG_IMPORTS = {
    'jf': '//production/borg/platforms-accelerators/jellyfish/jellyfish.borg',
    'df': '//production/borg/platforms-accelerators/jellyfish/dragonfish.borg'
}

_TPU_PLATFORM_KEY = {
    xm.Platform.JELLYFISH: 'jf',
    xm.Platform.DRAGONFISH: 'df',
}


def gpu_executable(model_dir):
  """Builds a GPU BuildTarget."""
  overrides = xm.BorgOverrides()
  overrides.xm_pass_arguments = False
  runtime = xm.Borg(
      cell=FLAGS.cell,
      priority=FLAGS.priority,
      overrides=overrides,
      # GPU machines have at least 10 CPUs per GPU: go/accelerator-resources-v2
      requirements=xm.Requirements(
          autopilot=False,
          cpu=10 * FLAGS.gpu_count,
          gpu=FLAGS.gpu_count,
          gpu_types=[FLAGS.gpu_type]),
  )
  return xm.BuildTarget(
      FLAGS.build_target_path,
      runtime=runtime,
      platform=xm.Platform.GPU,
      args=collections.OrderedDict([
          ('jax_port', '%port_jax%'),
          ('model_dir', model_dir),
          ('batch_size', FLAGS.batch_size),
          ('half_precision', FLAGS.half_precision),
          ('cache', True),
          ('report_dir', FLAGS.report_dir),
          ('resnet508b_ckpt_path', FLAGS.resnet508b_ckpt_path),
      ]),
  )


def tpu_chip_bounds(platform, topology, tpu_tasks_per_host):
  """Computes the number of chips per task in each topology axis."""
  if topology.chip_count == 1:
    if tpu_tasks_per_host != 1:
      raise ValueError(
          'A 1x1 job must have --tpu_tasks_per_host=1, got {}'.format(
              tpu_tasks_per_host))
    return [1, 1, 1]

  if platform == xm.Platform.JELLYFISH:
    if tpu_tasks_per_host != 1:
      raise ValueError('Jellyfish only supports --tpu_tasks_per_host=1')
    return [2, 2, 1]
  elif platform == xm.Platform.DRAGONFISH:
    if topology.chip_count > 4:
      df_pod_chip_bounds = {
          1: [4, 2, 1],
          2: [2, 2, 1],
          4: [1, 2, 1],
          8: [1, 1, 1],
      }
      try:
        return df_pod_chip_bounds[tpu_tasks_per_host]
      except KeyError:
        raise ValueError(
            'For a Dragonfish pod slice, --tpu_tasks_per_host must be one of '
            '1, 2, 4, or 8; got {}'.format(tpu_tasks_per_host))
    else:
      df_donut_chip_bounds = {
          1: [2, 2, 1],
          2: [1, 2, 1],
          4: [1, 1, 1],
      }
      try:
        return df_donut_chip_bounds[tpu_tasks_per_host]
      except KeyError:
        raise ValueError(
            'For a Dragonfish donut, --tpu_tasks_per_host must be one of '
            '1, 2, or 4; got {}'.format(tpu_tasks_per_host))
  else:
    raise ValueError('Unknown TPU platform {}'.format(platform))


def tpu_executable(platform, model_dir):
  """Builds a TPU BuildTarget."""
  topology = xm.TpuTopology(FLAGS.tpu_topology)

  tpu_tasks_per_host = FLAGS.tpu_tasks_per_host
  is_df_pod = platform == xm.Platform.DRAGONFISH and topology.chip_count > 4
  if tpu_tasks_per_host < 0:
    tpu_tasks_per_host = (2 if is_df_pod else 1)

  topology_dims = list(map(int, FLAGS.tpu_topology.split('x'))) + [1]
  chip_bounds = tpu_chip_bounds(platform, topology, tpu_tasks_per_host)
  host_bounds = [t // h for t, h in zip(topology_dims, chip_bounds)]

  overrides = xm.BorgOverrides()
  overrides.xm_pass_arguments = True
  if platform == xm.Platform.DRAGONFISH:
    overrides.tasks_per_host = tpu_tasks_per_host
    overrides.replicas = xm.borg_token("df.slices['{}'].hosts * {}".format(
        topology.topology_str, tpu_tasks_per_host))

  if FLAGS.tpu_topology == '4x2':
    # 4x2 requires the following scheduling overrides, per:
    # https://cs.corp.google.com/piper///depot/google3/production/borg/platforms-accelerators/jellyfish/dragonfish.borg?rcl=245509799&l=34
    overrides.scheduling.policy = 'BEST_EFFORT'
    overrides.scheduling.size = 1

  chips_per_task = functools.reduce(operator.mul, chip_bounds)
  # See go/accelerator-resources-v2
  cpu_per_chip = 13 if is_df_pod else 26
  host_ram_per_chip = 40 if is_df_pod else 80
  runtime = xm.Borg(
      cell=FLAGS.cell,
      priority=FLAGS.priority,
      overrides=overrides,
      requirements=xm.Requirements(
          autopilot=False,
          cpu=cpu_per_chip * chips_per_task,
          ram=host_ram_per_chip * chips_per_task * xm.GiB,
          topology=topology),
      imports=_TPU_BORG_CONFIG_IMPORTS,
      logs_read_access_roles=['all'],
  )

  mixin = xm.FragmentedPythonPackage(
      '//third_party/google_research/google_research/aqt/jax/imagenet:imagenet_fragmented_mpms'
  )

  return xm.Binary(
      name='imagenet',
      binary=mixin.launcher_path(),  # Returns a path to Python interpreter.
      runtime=runtime,
      mixins=[mixin],
      args=collections.OrderedDict([
          ('deepsea_chips_per_host_bounds', ','.join(map(str, chip_bounds))),
          ('deepsea_host_bounds', ','.join(map(str, host_bounds))),
          ('deepsea_mesh_controller_address',
           xm.borg_token('get_job_bns_prefix() + "/0:uberdriver"')),
          ('deepsea_mesh_controller_port', '%port_uberdriver%'),
          ('jax_port', '%port_jax%'),
          ('model_dir', model_dir),
          ('batch_size', FLAGS.batch_size),
          ('half_precision', FLAGS.half_precision),
          ('cache', True),
          ('report_dir', FLAGS.report_dir),
          ('resnet508b_ckpt_path', FLAGS.resnet508b_ckpt_path),
      ]),
      platform=platform,
  )


def main(_):
  # Key experiment details.
  model_dir = '/cns/{}-d/home/{}/flax_imagenet/{}_{}/'.format(
      FLAGS.cell, getpass.getuser(), FLAGS.name,
      datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))

  work_unit_dir = '{}%{}%/'.format(model_dir,
                                   xm_helper.experiment.work_unit_hyper_str)

  description = xm.ExperimentDescription(FLAGS.name)

  platform = xm.Platform.from_str(FLAGS.platform)
  is_tpu = xm.Platform.is_tpu(platform)

  if is_tpu:
    executable = tpu_executable(platform, work_unit_dir)
  else:
    executable = gpu_executable(work_unit_dir)

  hparams_config_filename = gpath.GPath(FLAGS.hparams_config_filename)
  if FLAGS.hparams_config_dir and not hparams_config_filename.is_absolute():
    hparams_config_filename = FLAGS.hparams_config_dir / hparams_config_filename

  experiment = xm_launch_lib.create_experiment(
      hparams_config_filename=hparams_config_filename,
      model_dir=model_dir,
      name=FLAGS.name,
      executable=executable)

  # Launch experiment on Borg.
  experiment_id = xm.launch_experiment(description, experiment)
  del experiment_id


if __name__ == '__main__':
  app.run()
