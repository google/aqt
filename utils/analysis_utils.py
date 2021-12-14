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

"""Collection of commonly used convenience functions for experiment analysis."""
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import dataclasses
import pandas as pd
import tree

from google3.third_party.google_research.google_research.aqt.utils import pandas_utils
from google3.third_party.google_research.google_research.aqt.utils import report_utils

# BEGIN GOOGLE-INTERNAL
from google3.pyglib.contrib.gpathlib import gpath
# END GOOGLE-INTERNAL


def flatten_with_joined_string_paths(
    dictionary: Dict[str, Any]) -> Dict[str, Any]:
  """Flattens nested dict to single level dict with joined paths as keys."""
  flattened = tree.flatten_with_path(structure=dictionary)
  flattened_dict = {}
  # join path tuples to single string
  for path_tuple, val in flattened:
    # convert all path elements to strings
    path = [str(s) for s in path_tuple]
    path = '/'.join(path)
    flattened_dict[path] = val
  return flattened_dict


def convert_report_to_flat_dict_default(
    report: report_utils.ExperimentReport) -> Dict[str, Any]:
  """Selects subset of report and flattens it to a single level dict.

  This function selects all information except what's stored under the fields
  `report_query_args` and `metadata_corp`.

  This function serves as an example for how to parse an ExperimentReport
  into a dataframe row by flattening it to a row_dict, with keys corresponding
  to dataframe columns.

  The ExperimentReport dataclass likely contains more information then you need
  for your analysis, so you can write your own function to pick and choose the
  information you want. You can refer to report_utils.ExperimentReport for
  documentation of all available fields.

  You can pass your custom function into convert_reports_to_dataframe().

  Args:
    report: An instance of ExperimentReport.

  Returns:
    A flattened dict representing a dataframe row.
  """

  row_dict = {}

  # Add smoothed metrics if present
  if report.metrics is not None:
    flattened_metrics = dict(flatten_with_joined_string_paths(report.metrics))
    # merge dicts
    row_dict = {**row_dict, **flattened_metrics}

  # Add unsmoothed metrics if present
  if report.unsmoothed_metrics is not None:
    flattened_unsmoothed_metrics = dict(
        flatten_with_joined_string_paths(report.unsmoothed_metrics))
    flattened_unsmoothed_metrics = {
        f'unsmoothed/{k}': v for k, v in flattened_unsmoothed_metrics.items()
    }
    # merge dicts
    row_dict = {**row_dict, **flattened_unsmoothed_metrics}

  # BEGIN GOOGLE-INTERNAL
  # Add XManager ID
  row_dict['xid'] = report.metadata_corp.xid

  # Add compute / memory cost information if present
  if report.compute_memory_cost is not None:
    row_dict['compute_cost'] = int(report.compute_memory_cost['compute_cost'])
    row_dict['memory_cost'] = int(report.compute_memory_cost['memory_cost'])
  # END GOOGLE-INTERNAL

  # Ignore following fields because they have already been added, or we chose
  # not to include them.
  report_fields_to_ignore = {
      'metrics',
      'unsmoothed_metrics',
      # BEGIN GOOGLE-INTERNAL
      'metadata_corp',
      'compute_memory_cost'
      # END GOOGLE-INTERNAL
  }
  # Add other report fields.
  for field in dataclasses.fields(report):
    if field.name not in report_fields_to_ignore:
      row_dict[field.name] = getattr(report, field.name)

  return row_dict


def convert_reports_to_dataframe(
    reports: List[report_utils.ExperimentReport],
    convert_report_to_flat_dict_fn: Callable[
        [report_utils.ExperimentReport],
        Dict[str, Any]] = convert_report_to_flat_dict_default
) -> pd.DataFrame:
  """Converts a list of ExperimentReport instances to a pandas dataframe.

  Args:
    reports: List of ExperimentReport instances. Each instance will correspond
      to a row in the dataframe.
    convert_report_to_flat_dict_fn: Function to use for converting an
      ExperimentReport to a flat dict, which will then be read in as a pandas
      dataframe row. The keys in the flat dict are interpreted as column names,
      the values as entries for that row. Please refer to
      `convert_report_to_flat_dict_default()` as an example.

  Returns:
    A pandas dataframe populated with information extracted from the reports.
  """

  rows = [convert_report_to_flat_dict_fn(rep) for rep in reports]
  return pd.DataFrame(rows)


def clickable_link(link: str, display_str: str = 'link') -> str:
  """Converts a link string into a clickable link with html tag.

  WARNING: This function is not safe to use for untrusted inputs since the
  generated HTML is not sanitized.

  Usage:
    df.style.format(clickable_link, subset=['col_name'])

  Args:
    link: A link string without formatting.
    display_str: What text the link should display.

  Returns:
    HTML-formatted link.

  """
  return f'<a href="{link}">{display_str}</a>'


# BEGIN GOOGLE-INTERNAL
def generate_diff_link(lhs_file_path: str, rhs_file_path: str) -> str:
  """Creates internal diff link, diffing the two files.

  Args:
    lhs_file_path: First file path, e.g. CNS path.
    rhs_file_path: Second file path, e.g. CNS path.

  Returns:
    Link to file diff view on internal ocean-diff-viewer tool.
  """

  def _escape_fwd_slashes(path: str):
    return path.replace('/', '%2F')

  lhs_file_path = _escape_fwd_slashes(lhs_file_path)
  rhs_file_path = _escape_fwd_slashes(rhs_file_path)
  link_prefix = 'https://ocean-diff-viewer.corp.google.com/text?'
  link = f'{link_prefix}lhs={lhs_file_path}&rhs={rhs_file_path}'
  return link


def convert_tensorboard_id_to_link(tensorboard_id: str) -> str:
  """Converts tensorboard ID to internal tensorboard link."""

  return f'https://tensorboard.corp.google.com/experiment/{tensorboard_id}/'


def get_tensorboard_link_for_experiments_in_df(df: pd.DataFrame) -> str:
  """Creates a tensorboard comparison link for experiments in dataframe.

  Args:
    df: A pandas dataframe with tensorboard id column.

  Returns:
    A tensorboard corp link.

  Raises:
    ValueError if dataframe does not have tensorboard_id column.

  """
  if 'tensorboard_id' not in list(df.columns):
    raise ValueError('Dataframe does not have tensorboard_id column.')
  tensorboard_ids = list(df.tensorboard_id.unique())
  tb_link = 'https://tensorboard.corp.google.com/compare/'
  for i, tid in enumerate(tensorboard_ids):
    if tid is not None:
      tb_link += f'{i}:{tid},'
    else:
      print('At least one row does not have a tensorboard id.')
  # Remove comma after last tensorboard id.
  if tb_link[-1] == ',':
    tb_link = tb_link[:-1]
  return tb_link


Regex = str
NewStr = str
ColumnName = str
ColumnValue = Any
OrderAscending = bool
SingleColumnFilter = Tuple[ColumnName, List[ColumnValue]]
SingleColumnRegex = Tuple[ColumnName, Regex]
SingleColumnRegexReplace = Tuple[ColumnName, Regex, Regex]
ColumnNameRegexReplace = Tuple[Regex, Regex]
SortBy = Sequence[Tuple[ColumnName, OrderAscending]]


def load_report_dataframe_with_filter_drop_rename(
    report_dir: gpath.GPath,
    row_filter_args: Optional[Sequence[SingleColumnFilter]] = None,
    row_regex_filter_args: Optional[Sequence[SingleColumnRegex]] = None,
    rename_row_value_args: Optional[Sequence[SingleColumnRegexReplace]] = None,
    drop_columns_by_regex_args: Optional[Sequence[Regex]] = None,
    rename_column_name_args: Optional[Sequence[ColumnNameRegexReplace]] = None,
    sort_by_args: Optional[SortBy] = None) -> pd.DataFrame:
  r"""Creates a report dataframe given directory.

  Optionally, the user can specify how to filter the dataframe rows, which
  columns to drop, and how to rename columns. For more customized pandas
  operations, please use report_utils.load_all_reports() to load all reports
  into a dataframe.

  Usage example:
    df = analysis_utils.load_report_dataframe_with_filter_drop_rename(
    report_dir=<dirpath>,
    row_filter_args=
      [('xid', [18925472, 18925219, 18925394])],
    row_regex_filter_args=
      [('model_dir', '.*4bit.*')],
    rename_row_value_args=
      [('experiment_name', r'leaderboard_full_model_(.*)_wanglisa-.*',
      r'\g<1>')],
    drop_columns_by_regex_args=['.*unsmoothed.*'],
    rename_column_name_args=[('_translate--de-en:test', '')],
    sort_by_args=([('eval/loss', True), ('eval/bleu', False)]),
  )

  Args:
    report_dir: Directory where report json files are stored.
    row_filter_args: List of row filters to be applied sequentially. Each row
      filter is a tuple of (column_name, filter_list). Each row filter is
      equivalent to a SQL statement of the form SELECT * WHERE column_name IN
      filter_list.
    row_regex_filter_args: Each row filter is a tuple of (column_name, regex).
      Will select the rows where column values match the regex.
    rename_row_value_args: Arguments for replacing values in columns. List of
      tuples (column_name, old_string, new_string).
    drop_columns_by_regex_args: Drop columns matching the provided regex.
    rename_column_name_args: Arguments to rename column headers. List of tuples
      (old_string (can be regex), new_string).
    sort_by_args: Arguments for df.sort_values(), List of tuples of (by,
      ascending), where `by` is a column sort by, and `ascending` is a bool
      indicating whether the column should be sorted in ascending order. First
      tuple will be used as primary sorting axis, etc. See documentation
      pandas.DataFrame.sort_values() for more details.

  Returns:
    A pandas dataframe.

  """

  reports = report_utils.load_all_reports(report_dir, num_threads=100)

  df = convert_reports_to_dataframe(reports,
                                    convert_report_to_flat_dict_default)

  df = pandas_utils.apply_filter_drop_rename_operations(
      df,
      row_filter_args=row_filter_args,
      row_regex_filter_args=row_regex_filter_args,
      rename_row_value_args=rename_row_value_args,
      drop_columns_by_regex_args=drop_columns_by_regex_args,
      rename_column_name_args=rename_column_name_args,
      sort_by_args=sort_by_args,
  )

  return df


# END GOOGLE-INTERNAL
