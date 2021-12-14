"""Util functions to generate an experiment report after training.

Please refer to the README.md for an overview of the reporting tool.
"""

import enum
import functools
import json
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import dataclasses
import numpy as onp

from google3.third_party.google_research.google_research.aqt.utils import tfevent_utils

# BEGIN GOOGLE-INTERNAL
from google3.googlex.positron.tensorflow.jax.shared_utils import hparams_utils
from google3.googlex.positron.tensorflow.reporting import utils as xm_query_utils
from google3.learning.deepmind.xmanager2.client import xmanager_api
from google3.pyglib import gfile
from google3.pyglib.concurrent import parallel
from google3.pyglib.contrib.gpathlib import gpath

# Constants
HPARAMS_CONFIG_FILENAME = gpath.GPath('hparams_config.json')
REPORT_FILENAME = gpath.GPath('report.json')
COMPUTE_MEMORY_COST_FILENAME = gpath.GPath('compute_memory_cost.json')

# END GOOGLE-INTERNAL

EventSeries = tfevent_utils.EventSeries

# Type Aliases
# Nested dict mapping from component (first key), attribute (second key) to
# events stored in EventSeries. E.g. component = 'train', attribute = 'loss'.
_AllEvents = Dict[str, Dict[str, EventSeries]]

# Nested dict mapping from component (first key), attribute (second key) to
# aggregated metric (float). E.g. component = 'train', attribute = 'loss'.
_AllAggMetrics = Dict[str, Dict[str, float]]


@enum.unique
class MinOrMax(enum.Enum):
  """Aggregation function to use for finding early stopping step."""
  MIN = enum.auto()  # use min value for early stopping step.
  MAX = enum.auto()  # use max value for early stopping step.

  def get_func(self) -> Callable[[onp.ndarray], int]:
    """Returns function associated with enum option. See parent class."""
    if self == MinOrMax.MIN:
      return onp.nanargmin
    elif self == MinOrMax.MAX:
      return onp.nanargmax
    else:
      raise ValueError('MinOrMax enum option not recognized.')


@enum.unique
class SmoothingKernel(enum.Enum):
  """Kernel function to use for smoothing."""
  # RECTANGULAR:Every value in symmetric window weighted equally. Values
  # outside the window are not included in average.
  # TRIANGULAR: Every value in symmetric window weighted as a linear function of
  # absolute distance to kernel center. Values outside the window are not
  # included in average.
  RECTANGULAR = enum.auto()
  TRIANGULAR = enum.auto()

  def rectangular_kernel(self, x: float, window_size_in_steps: int) -> float:
    """Rectangular kernel for moving window average.

    All values in window are equally weighted.

    Args:
      x: Distance to kernel center in steps.
      window_size_in_steps: Size of the window to average over.

    Returns:
      Unnormalized weight to use for averaging, e.g. in `np.average()`.

    Raises:
      ValueError: If window_size_in_steps arg is less than 1.
    """
    if window_size_in_steps < 1:
      raise ValueError('window_size_in_steps has to be >= 1.')
    if abs(x) <= window_size_in_steps / 2:
      return 1.0
    else:
      return 0.0

  def triangular_kernel(self, x: float, window_size_in_steps: int) -> float:
    """Triangular kernel for moving window average.

    The weight is a linear function of the absolute distance to the kernel
    center.

    Args:
      x: Distance to kernel center in steps.
      window_size_in_steps: Size of the window to average over.

    Returns:
      Unnormalized weight to use for averaging, e.g. in `np.average()`.

    Raises:
      ValueError: If window_size_in_steps arg is less than 1.
    """
    if window_size_in_steps < 1:
      raise ValueError('window_size_in_steps has to be >= 1.')
    return max(0.0, window_size_in_steps / 2 - abs(x))

  def get_func(
      self,
      window_size_in_steps: Optional[int] = None) -> Callable[[float], float]:
    """Returns function associated with enum option. See parent class."""
    if self == SmoothingKernel.RECTANGULAR:
      if window_size_in_steps is None:
        raise ValueError('For rectangular smoothing_kernel '
                         'window_size_in_steps must be provided.')
      return functools.partial(
          self.rectangular_kernel, window_size_in_steps=window_size_in_steps)
    elif self == SmoothingKernel.TRIANGULAR:
      if window_size_in_steps is None:
        raise ValueError('For triangular smoothing_kernel '
                         'window_size_in_steps must be provided.')
      return functools.partial(
          self.triangular_kernel, window_size_in_steps=window_size_in_steps)
    else:
      raise ValueError('SmoothingKernel enum option not recognized.')


# BEGIN GOOGLE-INTERNAL
@dataclasses.dataclass
class MetadataCorp:
  """Google-internal experiment metadata."""

  # XManager experiment ID.
  xid: Optional[int] = None

  # XManager worker ID.
  wid: Optional[int] = None

  # Information about the citc client from which experiment was launched.
  citc_client_info: Optional[xm_query_utils.XManagerClientInfo] = None
# END GOOGLE-INTERNAL


@dataclasses.dataclass
class ExperimentReport:
  """Report for a single experiment run based on its TFEvents files."""
  # Model directory corresponding to single run, with TFEvents files to
  # generate report from.
  model_dir: str

  # Metrics at early stop step, with smoothing applied.
  # If NaN values present, then this field will
  # be left None, but unsmoothed_metrics will still be reported.
  # maps component name (e.g. eval) to metrics dict, which in turn maps
  # attribute name to scalar value.
  metrics: Optional[_AllAggMetrics]

  # Metrics without smoothing at early stop step.
  # maps component name (e.g. eval) to metrics dict, which in turn maps
  # attribute name to scalar value.
  unsmoothed_metrics: Optional[_AllAggMetrics]

  # Step at which early_stop_attr in early_stop_ds_dir is minimized. Scalars are
  # reported at this step.
  early_stop_step: int

  # Number of training steps. In combination with early_stop_step, can help
  # determine whether training converged and started to overfit.
  num_train_steps: int

  # Arguments passed into create_end_of_training_report(), the function that
  # created this report.
  # Included here because the arguments can impact the reported metrics, e.g.
  # which attribute was used to find the early stopping step.
  report_query_args: Dict[str, Any]

  # Human-readable experiment name.
  experiment_name: Optional[str] = None

  # Name of user who launched the experiment.
  user_name: Optional[str] = None

  # When experiment was launched, formatted as '%Y%m%dT%H%M%S'.
  launch_time: Optional[str] = None

  # Evaluation frequency. How often summaries were saved to file.
  eval_freq: Optional[int] = None

  # If any metrics contain NaN values, first step at which a NaN value occurs.
  first_nan_step: Optional[int] = None

  # Tensorboard ID or URL.
  tensorboard_id: Optional[str] = None

  # BEGIN GOOGLE-INTERNAL
  # Full path to hparams config file.
  hparams_config_path: Optional[str] = None

  # Compute and memory cost information, for more details see
  # googlex/positron/tensorflow/jax/shared_utils/compute_cost_utils.py.
  compute_memory_cost: Optional[Dict[str, float]] = None

  # Google-specific metadata (XManager, CitC etc.)
  metadata_corp: Optional[MetadataCorp] = None
  # END GOOGLE-INTERNAL


def check_for_nans(event_series: EventSeries, start_step: int) -> Optional[int]:
  """Finds step >= start_step at which first NaN value occurs if there are any.

  Args:
    event_series: list of tuples (step, value).
    start_step: After which step to check for NaNs.

  Returns:
    Step at which first NaN value occurs, or None otherwise.

  """
  keep_indices = (event_series.steps >= start_step)
  event_series.steps = event_series.steps[keep_indices]
  event_series.values = event_series.values[keep_indices]
  nan_indices = onp.argwhere(onp.isnan(event_series.values))
  if nan_indices.size:
    return int(event_series.steps[onp.min(nan_indices)])
  return None


def check_all_events_for_nans(all_events: _AllEvents) -> Optional[int]:
  """Finds step at which first NaN value occurs if there are any.

  Args:
    all_events: Nested dict mapping from component, attribute, to EventSeries.

  Returns:
    Step at which first NaN value occurs, or None otherwise.
  """

  first_nan_step = None

  for events_dict in all_events.values():
    for events in events_dict.values():
      cur_first_nan_step = check_for_nans(events, start_step=0)
      if cur_first_nan_step is None:
        continue
      if first_nan_step is None:
        first_nan_step = cur_first_nan_step
      else:
        first_nan_step = min(first_nan_step, cur_first_nan_step)
  return first_nan_step


def find_early_stop_step(event_series: EventSeries,
                         early_stop_func: Callable[[onp.ndarray], int],
                         start_step: int) -> int:
  """Finds step >= start_step at which event_series is minimized.

  Args:
    event_series: list of tuples (step, value).
    early_stop_func: Aggregator function to use to find early_stop_step.
    start_step: After which step to include values in moving average.

  Returns:
    Step at which moving average of series is minimized.

  """
  keep_indices = (event_series.steps >= start_step)
  event_series.steps = event_series.steps[keep_indices]
  event_series.values = event_series.values[keep_indices]

  if event_series.steps.size == 0:
    raise ValueError('event_series does not have events after start_step.')

  if onp.all(onp.isnan(event_series.values)):
    return start_step
  early_stop_idx = early_stop_func(event_series.values)
  return int(event_series.steps[early_stop_idx])


def apply_smoothing_about_step(events: EventSeries, step: int,
                               kernel_fn: Callable[[float], float]) -> float:
  """Applies smoothing of event values for a single step.

  Args:
    events: list of tuples (step, value).
    step: Step to apply smoothing about.
    kernel_fn: Kernel function to use for smoothing.

  Returns:
    Smoothed value at step.

  Raises:
    ValueError: If NaN values present in events.values.
  """
  if check_for_nans(events, start_step=0) is not None:
    raise ValueError(
        'NaN values encountered in smoothing, which is not supported.')
  weights = onp.vectorize(kernel_fn)(events.steps - step)
  return float(onp.average(events.values, weights=weights))


def apply_smoothing(events: EventSeries,
                    kernel_fn: Callable[[float], float]) -> EventSeries:
  """Applies smoothing of event values over all steps.

  Args:
    events: list of tuples (step, value).
    kernel_fn: Kernel function to use for smoothing.

  Returns:
    Smoothed events for all steps in steps arg.
  """
  smoothed_events = EventSeries(
      name=events.name,
      steps=onp.array([], dtype=int),
      values=onp.array([]),
      wall_times=None)
  for i in range(len(events.steps)):
    smoothed_events.steps = onp.append(smoothed_events.steps, events.steps[i])
    smoothed_value = apply_smoothing_about_step(
        events=events, step=events.steps[i], kernel_fn=kernel_fn)
    smoothed_events.values = onp.append(smoothed_events.values, smoothed_value)

  return smoothed_events


def get_agg_metrics_at_step(
    all_events: _AllEvents, step: int,
    smoothing_kernel_fn: Optional[Callable[[float], float]]) -> _AllAggMetrics:
  """Computes aggregated metrics from EventSeries dicts at early stop step.

  Args:
    all_events: Nested dict mapping from component, attribute, to EventSeries.
    step: Step at which to get event values to compute aggregated metrics.
    smoothing_kernel_fn: If None, no smoothing will be applied. If any NaNs are
      present, has to be set to None, otherwise ValueError will be raised.

  Returns:
    dict mapping from (component, attribute) to aggregated scalar metric.

  """
  all_agg_metrics = {}
  for component, events_dict in all_events.items():
    agg_metrics_dict = {}
    for attr, events in events_dict.items():
      if smoothing_kernel_fn is None:
        index = onp.argmin(onp.abs(events.steps - step))
        agg_metrics_dict[attr] = events.values[index]
      else:
        agg_metrics_dict[attr] = apply_smoothing_about_step(
            events, step=step, kernel_fn=smoothing_kernel_fn)
    all_agg_metrics[str(component)] = agg_metrics_dict

  return all_agg_metrics


def compute_agg_metrics_from_events(
    all_events: _AllEvents,
    early_stop_component: str,
    early_stop_attr: str,
    early_stop_agg: MinOrMax,
    smoothing_kernel: SmoothingKernel,
    window_size_in_steps: Optional[int] = None,
    start_step: int = 0
) -> Tuple[_AllAggMetrics, Optional[_AllAggMetrics], int, Optional[int]]:
  """Computes aggregated metrics from EventSeries dicts.

  Args:
    all_events: Nested dict mapping from component, attribute, to EventSeries.
    early_stop_component: Which component to use to find early_stop_step.
    early_stop_attr: Attribute to find minimum or maximum of, e.g. 'perplexity'.
    early_stop_agg: Which aggregator to use to find early_stop_step. See
      MinOrMax class for enum options.
    smoothing_kernel: Which kernel to use for smoothing. See SmoothingKernel
      class for enum options.
    window_size_in_steps: Only applicable to some kernels, including
      'rectangular' kernel. Number of steps to average over.
    start_step: After which step to consider early stopping, e.g. if set to 100,
      only steps >= 100 will be considered.

  Returns:
    Tuple of dict mapping from (component, attribute) to aggregated scalar
      metric and early_stop_step.

  """
  first_nan_step = check_all_events_for_nans(all_events=all_events)

  early_stop_func = early_stop_agg.get_func()
  early_stop_events = all_events[early_stop_component][early_stop_attr]

  if first_nan_step is None:
    # Apply smoothing to early stop component events.
    smoothing_kernel_func = smoothing_kernel.get_func(
        window_size_in_steps=window_size_in_steps)
    early_stop_events = apply_smoothing(
        events=early_stop_events, kernel_fn=smoothing_kernel_func)

  early_stop_step = find_early_stop_step(
      early_stop_events, early_stop_func=early_stop_func, start_step=start_step)

  all_agg_metrics_unsmoothed = get_agg_metrics_at_step(
      all_events=all_events, step=early_stop_step, smoothing_kernel_fn=None)

  if first_nan_step is None:
    # Only get smoothed metrics if no NaN values found.
    all_metrics_smoothed = get_agg_metrics_at_step(
        all_events=all_events,
        step=early_stop_step,
        smoothing_kernel_fn=smoothing_kernel_func)
  else:
    all_metrics_smoothed = None

  return all_agg_metrics_unsmoothed, all_metrics_smoothed, early_stop_step, first_nan_step


def create_end_of_training_report_oss(
    model_dir: str,
    eval_freq: int,
    num_train_steps: int,
    early_stop_attr: str,
    early_stop_agg: MinOrMax,
    smoothing_kernel: SmoothingKernel,
    early_stop_ds_dir: Optional[str] = None,
    other_ds_dirs: Optional[List[str]] = None,
    tags_to_include: Optional[List[str]] = None,
    window_size_in_steps: int = 1,
    start_step: int = 0,
    experiment_name: Optional[str] = None,
    user_name: Optional[str] = None,
    launch_time: Optional[str] = None,
    tensorboard_id: Optional[str] = None,
) -> ExperimentReport:
  """Creates an experiment report from TFEvents data after training completion.

  Args:
    model_dir: A model directory corresponding to a single model run, with
      TFEvent file(s) and a single hparams_config file. The TFEvent files can
      either be stored directly in model_dir, or in subdirectories in model_dir,
      but not both.
    eval_freq: Frequency of event saving.
    num_train_steps: Number of training steps.
    early_stop_attr: Attribute to find minimum or maximum of, e.g. 'perplexity'.
    early_stop_agg: Which aggregator to use to find early_stop_step. See
      MinOrMax class for enum options.
    smoothing_kernel: Which kernel to use for smoothing. See SmoothingKernel
      class for enum options.
    early_stop_ds_dir: The events subdir in model_dir to use to find
      early_stop_step if model_dir has subdirs. The early_stop_attr within
      early_stop_ds_dir will be used to find the early_stop_step.
    other_ds_dirs: List of other subdirs in model_dir with events to report.
    tags_to_include: List of event tags that should be included.
    window_size_in_steps: Number of steps to average over. Should be multiple of
      eval_freq. If set to 1, no averaging will be applied.
    start_step: After which step to consider early stopping, e.g. if set to 100,
      only steps >= 100 will be considered.
    experiment_name:  Human-readable experiment name.
    user_name: Name of user who launched the experiment.
    launch_time: When experiment was launched, formatted as '%Y%m%dT%H%M%S'.
    tensorboard_id: Tensorboard ID, e.g. URL to tensorboard dev, if applicable.

  Returns:
    An ExperimentReport dataclass instance.

  """
  # Saving report query args, to be included in the report.
  report_query_args = {
      'early_stop_attr': early_stop_attr,
      'early_stop_agg': early_stop_agg.name,
      'early_stop_ds_dir': early_stop_ds_dir,
      'other_ds_dirs': other_ds_dirs,
      'tags_to_include': tags_to_include,
      'smoothing_kernel': smoothing_kernel.name,
      'window_size_in_steps': window_size_in_steps,
      'start_step': start_step,
  }

  all_events = {}

  early_stop_component = None

  # If subdirs provided
  if early_stop_ds_dir is not None or other_ds_dirs is not None:
    if early_stop_ds_dir is None:
      raise ValueError(
          'If other_ds_dirs is not None, early_stop_ds_dir has to be '
          'provided.')

    early_stop_events = tfevent_utils.get_parsed_tfevents(
        os.path.join(model_dir, early_stop_ds_dir), tags_to_include)
    early_stop_component = early_stop_ds_dir
    all_events[early_stop_component] = early_stop_events

    if other_ds_dirs is not None:
      for ds_dir in other_ds_dirs:
        if ds_dir is not None:
          all_events[ds_dir] = tfevent_utils.get_parsed_tfevents(
              os.path.join(model_dir, ds_dir), tags_to_include)
  else:
    # If no subdirs provided, will assume that there are no subcomponents.
    # For consistency with the case when we do have components, we store the
    # events under the dummy component 'all'.
    early_stop_component = 'all'
    all_events[early_stop_component] = tfevent_utils.get_parsed_tfevents(
        model_dir, tags_to_include)

  all_agg_metrics_unsmoothed, all_agg_metrics_smoothed, early_stop_step, first_nan_step = compute_agg_metrics_from_events(
      all_events=all_events,
      early_stop_component=early_stop_component,
      early_stop_attr=early_stop_attr,
      early_stop_agg=early_stop_agg,
      smoothing_kernel=smoothing_kernel,
      window_size_in_steps=window_size_in_steps,
      start_step=start_step)

  report = ExperimentReport(
      model_dir=model_dir,
      metrics=all_agg_metrics_smoothed,
      unsmoothed_metrics=all_agg_metrics_unsmoothed,
      early_stop_step=early_stop_step,
      num_train_steps=num_train_steps,
      eval_freq=eval_freq,
      first_nan_step=first_nan_step,
      experiment_name=experiment_name,
      user_name=user_name,
      launch_time=launch_time,
      tensorboard_id=tensorboard_id,
      report_query_args=report_query_args,
  )

  return report


# BEGIN GOOGLE-INTERNAL
def create_end_of_training_report(
    model_dir: gpath.GPath,
    xid: int,
    eval_freq: int,
    num_train_steps: int,
    early_stop_attr: str,
    early_stop_agg: MinOrMax,
    smoothing_kernel: SmoothingKernel,
    early_stop_ds_dir: Optional[gpath.GPath] = None,
    other_ds_dirs: Optional[List[gpath.GPath]] = None,
    tags_to_include: Optional[List[str]] = None,
    window_size_in_steps: int = 1,
    start_step: int = 0,
    wid: Optional[int] = None,
) -> ExperimentReport:
  """Google-internal wrapper function of create_end_of_training_report().

  Pulls additional information from XManager experiment to populate report.

  Args:
    model_dir: A model directory corresponding to a single model run, with
      TFEvent file(s) and a single hparams_config file. The TFEvent files can
      either be stored directly in model_dir, or in subdirectories in model_dir,
      but not both.
    xid: XManager experiment id. Used to retrieve metadata.
    eval_freq: Frequency of event saving.
    num_train_steps: Number of training steps.
    early_stop_attr: Attribute to find minimum or maximum of, e.g. 'perplexity'.
    early_stop_agg: Which aggregator to use to find early_stop_step. See
      MinOrMax class for enum options.
    smoothing_kernel: Which kernel to use for smoothing. See SmoothingKernel
      class for enum options.
    early_stop_ds_dir: The events subdir in model_dir to use to find
      early_stop_step if model_dir has subdirs. The early_stop_attr within
      early_stop_ds_dir will be used to find the early_stop_step.
    other_ds_dirs: List of other subdirs in model_dir with events to report.
    tags_to_include: List of event tags that should be included.
    window_size_in_steps: Number of steps to average over. Should be multiple of
      eval_freq. If set to 1, no averaging will be applied.
    start_step: After which step to consider early stopping, e.g. if set to 100,
      only steps >= 100 will be considered.
    wid: XManager worker id.

  Returns:
    An ExperimentReport dataclass instance.

  """

  xm_client = xmanager_api.XManagerApi(xm_deployment_env='alphabet')
  experiment = xm_client.get_experiment(xid)

  experiment_name = experiment.name
  user_name = experiment.author
  launch_time = experiment.creation_time.strftime('%Y%m%dT%H%M%S')

  xm_query = xm_query_utils.XManagerQuery([xid])
  tensorboard_id = str(xm_query.xmanager_to_mldash_mapping()[xid])

  if early_stop_ds_dir is not None:
    early_stop_ds_dir = str(early_stop_ds_dir)
  if other_ds_dirs is not None:
    other_ds_dirs = [str(d) for d in other_ds_dirs]

  report = create_end_of_training_report_oss(
      model_dir=str(model_dir),
      eval_freq=eval_freq,
      num_train_steps=num_train_steps,
      early_stop_attr=early_stop_attr,
      early_stop_agg=early_stop_agg,
      smoothing_kernel=smoothing_kernel,
      early_stop_ds_dir=early_stop_ds_dir,
      other_ds_dirs=other_ds_dirs,
      tags_to_include=tags_to_include,
      window_size_in_steps=window_size_in_steps,
      start_step=start_step,
      experiment_name=experiment_name,
      user_name=user_name,
      launch_time=launch_time,
      tensorboard_id=tensorboard_id,
  )

  compute_memory_cost = None
  if str(COMPUTE_MEMORY_COST_FILENAME) in gfile.ListDir(model_dir):
    with gfile.Open(model_dir / COMPUTE_MEMORY_COST_FILENAME, 'rt') as f:
      compute_memory_cost = json.load(f)

  metadata_corp = MetadataCorp(
      xid=xid,
      wid=wid,
      citc_client_info=xm_query.xmanager_client_infos()[xid],
  )

  report.metadata_corp = metadata_corp
  report.compute_memory_cost = compute_memory_cost
  report.hparams_config_path = str(model_dir / HPARAMS_CONFIG_FILENAME)

  return report


def report_path_from_model_dir(model_dir: gpath.GPath) -> gpath.GPath:
  """Generates report path from model_dir."""
  report_path = '--'.join(list(model_dir.parts)[1:])

  # get rid of file extensions if there were any
  report_path = os.path.splitext(str(report_path))[0]

  return gpath.GPath(report_path)


def save_report(report: ExperimentReport,
                report_dir: gpath.GPath) -> gpath.GPath:
  """Saves json serialized ExperimentReport instance to report directory.

  If report has hparams_config_path, will attempt to copy over the config file
  to the report_dir as well.

  The filename is auto-generated by concatenating report.model_dir.

  Args:
    report: The report to be saved.
    report_dir: The directory to save to.

  Returns:
    Path where report was saved to.
  """
  report_path = report_path_from_model_dir(gpath.GPath(report.model_dir))
  if not gfile.Exists(report_dir / report_path):
    gfile.MkDir(report_dir / report_path)
  if report.hparams_config_path is not None:
    gfile.Copy(
        report.hparams_config_path,
        report_dir / report_path / HPARAMS_CONFIG_FILENAME,
        overwrite=True)
    report.hparams_config_path = str(report_dir / report_path /
                                     HPARAMS_CONFIG_FILENAME)
  hparams_utils.save_dataclass_to_disk(
      report, report_dir / report_path / REPORT_FILENAME)
  return report_dir / report_path


def load_report_fn(report_path: gpath.GPath) -> Optional[ExperimentReport]:
  """Tries to load report from path, if it exists. Returns None otherwise."""
  report = None
  if not gfile.Exists(report_path):
    print(f'File {report_path} does not exist.')
    return report

  try:
    report = hparams_utils.load_dataclass_from_disk(ExperimentReport,
                                                    report_path)
  except gfile.FileError as file_error:
    print(f'Failed to load file {report_path}.')
    print(file_error)
  return report


def load_report(model_dir: gpath.GPath,
                report_dir: gpath.GPath) -> Optional[ExperimentReport]:
  """Loads existing ExperimentReport instance from report directory.

  Args:
    model_dir: Model directory of run to load report for (lookup key).
    report_dir: Report directory in which reports are saved.

  Returns:
    An ExperimentReport instance.
  """
  model_report_path = report_path_from_model_dir(model_dir)
  report_path = report_dir / model_report_path / REPORT_FILENAME
  return load_report_fn(report_path)


def load_all_reports(report_dir: gpath.GPath,
                     num_threads: int) -> List[ExperimentReport]:
  """Loads all reports from report directory.

  Args:
    report_dir: Report directory in which reports are saved.  Returns
      ExperimentReport instance.
    num_threads: Number of threads (threadpool workers) for parallelization.

  Returns:
    A list of ExperimentReport instances.
  """
  list_of_kwargs_to_function = [{
      'report_path': report / REPORT_FILENAME
  } for report in report_dir.iterdir()]
  start = time.time()
  reports = parallel.RunInParallel(
      function=load_report_fn,
      list_of_kwargs_to_function=list_of_kwargs_to_function,
      num_workers=num_threads)
  # remove None values from reports list
  reports = [rep for rep in reports if rep]
  end = time.time()
  print(f'time to load reports: {end-start}')
  print(f'Number of reports loaded: {len(reports)}')

  return reports
# END GOOGLE-INTERNAL
