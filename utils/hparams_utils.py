"""Functions to load/save the hparams to/from a config dict."""

import json
import os
import typing
from typing import Any, Dict, Optional, Type, TypeVar

import dacite
import dataclasses
import jax
import ml_collections

# BEGIN GOOGLE-INTERNAL
from google3.pyglib import gfile
from google3.pyglib.contrib.gpathlib import gpath
# END GOOGLE-INTERNAL

from google3.third_party.google_research.google_research.aqt.jax import quant_config
from google3.third_party.google_research.google_research.aqt.jax import quantization
from google3.third_party.google_research.google_research.aqt.jax.flax import struct as flax_struct

T = TypeVar('T')

dataclass = flax_struct.dataclass if not typing.TYPE_CHECKING else dataclasses.dataclass


@dataclass
class HParamsMetadata:
  """Metadata associated with an experiment configuration."""

  # Human-readable description of this hparams configuration. Mainly
  # useful for hand inspection of serialized JSON files.
  description: str

  # Creation time of the configuration in the format of seconds from epoch.
  # Used for versioning different hyperparameter settings for the same
  # model configuration.
  last_updated_time: Optional[float]

  # By default, it is used to name the model directory and label the
  # experiment in tensorboard.
  hyper_str: Optional[str] = None


# TODO(abdolrashidi): Add unit tests for the functions below.
def save_dataclass_to_disk(data, path: str):
  """Serializes the given dataclass to a JSON file on disk.

  Args:
    data: A dataclass instance.
    path: Path to save the dataclass to.
  """

  data_dict = dataclasses.asdict(data)
  with open(path, 'w') as file:
    json.dump(data_dict, file, indent=2)


def write_hparams_to_file_with_host_id_check(hparams,
                                             output_dir: Optional[str]):
  """Writes hparams to file for master host.

  Args:
    hparams: Hparams.
    output_dir: Output directory to save hparams to, saves as output_dir /
      'hparams_config.json.
  """
  if jax.host_id() == 0 and output_dir is not None:
    # The directory is usually created automatically by the time we reach here,
    # but on some training runs it appears not to be.
    # MakeDirs will create the directory if it doesn't already exist and is a
    # no-op if it already exists.

    # BEGIN GOOGLE-INTERNAL
    output_dir = gpath.GPath(output_dir)
    gfile.MakeDirs(output_dir)
    data_dict = dataclasses.asdict(hparams)
    with gfile.Open(output_dir / 'hparams_config.json', 'w') as file:
      json.dump(data_dict, file, indent=2)
    return
    # pylint: disable=unreachable
    # END GOOGLE-INTERNAL

    os.makedirs(output_dir, exist_ok=True)
    save_dataclass_to_disk(hparams,
                           os.path.join(output_dir, 'hparams_config.json'))


def load_dataclass_from_dict(dataclass_name: Type[T],
                             data_dict: Dict[Any, Any]) -> T:
  """Converts parsed dictionary from JSON into a dataclass.

  Args:
    dataclass_name: Name of the dataclass.
    data_dict: Dictionary parsed from JSON.

  Returns:
    An instance of `dataclass` populated with the data from `data_dict`.
  """
  # Some fields in TrainingHParams are formal Python enums, but they are stored
  # as plain text in the json. Dacite needs to be given a list of which classes
  # to convert from a string into an enum. The classes of all enum values which
  # are stored in a TrainingHParams instance (directly or indirectly) should be
  # listed here. See https://github.com/konradhalas/dacite#casting.
  enum_classes = [
      quantization.QuantOps.ActHParams.InputDistribution,
      quantization.QuantType, quant_config.QuantGranularity
  ]
  data_dict = _convert_lists_to_tuples(data_dict)
  return dacite.from_dict(
      data_class=dataclass_name,
      data=data_dict,
      config=dacite.Config(cast=enum_classes))


T = TypeVar('T')


def _convert_lists_to_tuples(node: T) -> T:
  """Recursively converts all lists to tuples in a nested structure.

  Recurses into all lists and dictionary values referenced by 'node',
  converting all lists to tuples.

  Args:
    node: A Python structure corresponding to JSON (a dictionary, a list,
      scalars, and compositions thereof)

  Returns:
    A Python structure identical to the input, but with lists replaced by
      tuples.
  """

  if isinstance(node, dict):
    return {key: _convert_lists_to_tuples(value) for key, value in node.items()}
  elif isinstance(node, (list, tuple)):
    return tuple([_convert_lists_to_tuples(value) for value in node])
  else:
    return node


def load_dataclass_from_json(dataclass_name: Type[T], json_data: str) -> T:
  """Creates a dataclass instance from JSON.

  Args:
    dataclass_name: Name of the dataclass to deserialize the JSON into.
    json_data: A Python string containing JSON.

  Returns:
    An instance of 'dataclass' populated with the JSON data.
  """

  data_dict = json.loads(json_data)
  return load_dataclass_from_dict(dataclass_name, data_dict)


# TODO(shivaniagrawal): functionality `load_hparams_from_file` is created for a
# generic (model hparams independent) train_hparams class; either we should move
# towards shared TrainHparams or remove the following functionalities.
def load_hparams_from_config_dict(hparams_classname: Type[T],
                                  model_classname: Type[Any],
                                  config_dict: ml_collections.ConfigDict) -> T:
  """Loads hparams from a configdict, and populates its model object.

  Args:
    hparams_classname: Name of the hparams class.
    model_classname: Name of the model class within the hparams class
    config_dict: A config dict mirroring the structure of hparams.

  Returns:
    An instance of 'hparams_classname' populated with the data from
    'config_dict'.
  """

  hparams = load_dataclass_from_config_dict(hparams_classname, config_dict)
  hparams.model_hparams = load_dataclass_from_dict(model_classname,
                                                   hparams.model_hparams)
  return hparams


def load_dataclass_from_config_dict(
    dataclass_name: Type[T], config_dict: ml_collections.ConfigDict) -> T:
  """Creates a dataclass instance from a configdict.

  Args:
    dataclass_name: Name of the dataclass to deserialize the configdict into.
    config_dict: A config dict mirroring the structure of 'dataclass_name'.

  Returns:
    An instance of 'dataclass_name' populated with the data from 'config_dict'.
  """

  # We convert the config dicts to JSON instead of a dictionary to force all
  # recursive field references to fully resolve in a way that Dacite can
  # consume.
  json_data = config_dict.to_json()
  return load_dataclass_from_json(dataclass_name, json_data)
