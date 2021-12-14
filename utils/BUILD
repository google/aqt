load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")
load("//devtools/python/blaze:strict.bzl", "py_strict_test")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

pytype_strict_library(
    name = "config_schema_utils",
    srcs = ["config_schema_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/ml_collections",
        "//third_party/py/ml_collections/config_dict",
    ],
)

py_strict_test(
    name = "config_schema_utils_test",
    srcs = ["config_schema_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":config_schema_utils",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/ml_collections",
    ],
)

pytype_strict_library(
    name = "report_utils",
    srcs = ["report_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":tfevent_utils",
        # BEGIN GOOGLE-INTERNAL
        "//googlex/positron/tensorflow/jax/shared_utils:hparams_utils",
        "//googlex/positron/tensorflow/reporting:utils",
        "//learning/deepmind/xmanager2/client:xmanager_api",
        "//pyglib:gfile",
        "//pyglib/concurrent:parallel",
        "//pyglib/contrib/gpathlib:gpath",
        # END GOOGLE-INTERNAL
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "tfevent_utils",
    srcs = ["tfevent_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "analysis_utils",
    srcs = ["analysis_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":pandas_utils",
        ":report_utils",
        # BEGIN GOOGLE-INTERNAL
        "//pyglib/contrib/gpathlib:gpath",
        # END GOOGLE-INTERNAL
        "//third_party/py/pandas",
        "//third_party/py/tree",
    ],
)

pytype_strict_library(
    name = "pandas_utils",
    srcs = ["pandas_utils.py"],
    srcs_version = "PY3",
    deps = ["//third_party/py/pandas"],
)

pytype_strict_library(
    name = "hparams_utils",
    srcs = ["hparams_utils.py"],
    srcs_version = "PY3",
    deps = [
        # BEGIN GOOGLE-INTERNAL
        "//pyglib:gfile",
        "//pyglib/contrib/gpathlib:gpath",
        # END GOOGLE-INTERNAL
        "//third_party/google_research/google_research/aqt/jax:quant_config",
        "//third_party/google_research/google_research/aqt/jax:quantization",
        "//third_party/google_research/google_research/aqt/jax/flax:struct",
        "//third_party/py/dacite",
        "//third_party/py/jax",
        "//third_party/py/ml_collections",
    ],
)

py_strict_test(
    name = "report_utils_test",
    srcs = ["report_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":report_utils",
        # BEGIN GOOGLE-INTERNAL
        "//pyglib/contrib/gpathlib:gpath",
        # END GOOGLE-INTERNAL
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "tfevent_utils_test",
    srcs = ["tfevent_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tfevent_utils",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "analysis_utils_test",
    srcs = ["analysis_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":analysis_utils",
        ":report_utils",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/pandas",
    ],
)

py_strict_test(
    name = "pandas_utils_test",
    srcs = ["pandas_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":pandas_utils",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/pandas",
    ],
)

pytype_strict_library(
    name = "summary_utils",
    srcs = [
        "summary_utils.py",
    ],
    deps = [
        "//third_party/py/flax/metrics:tensorboard",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "summary_utils_test",
    srcs = ["summary_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":summary_utils",
        "//third_party/google_research/google_research/aqt/jax:stats",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/jax",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "common",
    srcs = [
        "common.py",
    ],
)
