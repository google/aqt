# BUILD files for quantization libraries implemented in jax with flax.

load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")
load("//devtools/python/blaze:strict.bzl", "py_strict_test")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

pytype_strict_library(
    name = "fp_cast",
    srcs = ["fp_cast.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/jax",
    ],
)

py_strict_test(
    name = "fp_cast_test",
    srcs = ["fp_cast_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":fp_cast",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/jax",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "primitives_test",
    srcs = ["primitives_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":primitives",
        ":test_utils",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/jax",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "flax_attention_test",
    timeout = "long",
    srcs = ["flax_attention_test.py"],
    python_version = "PY3",
    shard_count = 6,
    srcs_version = "PY3",
    deps = [
        ":flax_attention",
        ":flax_layers",
        ":get_bounds",
        ":quant_config",
        ":quantization",
        ":test_utils",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/flax:core",
        "//third_party/py/jax",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "quantization_test",
    srcs = ["quantization_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":fp_cast",
        ":get_bounds",
        ":primitives",
        ":quant_config",
        ":quantization",
        ":test_utils",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/flax:core",
        "//third_party/py/jax",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "get_bounds_test",
    srcs = ["get_bounds_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":get_bounds",
        ":quant_config",
        ":test_utils",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/flax",
        "//third_party/py/jax",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "flax_layers_test",
    timeout = "long",
    srcs = ["flax_layers_test.py"],
    python_version = "PY3",
    shard_count = 10,
    srcs_version = "PY3",
    deps = [
        ":flax_layers",
        ":fp_cast",
        ":get_bounds",
        ":primitives",
        ":quant_config",
        ":quantization",
        ":shape_utils",
        ":test_utils",
        "//third_party/py/absl/flags",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/flax:core",
        "//third_party/py/jax",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "stats_test",
    srcs = ["stats_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":stats",
        ":test_utils",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/jax",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "stats_tag_test",
    srcs = ["stats_tag_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":stats_tag",
        ":test_utils",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/jax",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":quantization",
        ":stats",
        "//third_party/py/jax",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "primitives",
    srcs = ["primitives.py"],
    srcs_version = "PY3",
    deps = ["//third_party/py/jax"],
)

pytype_strict_library(
    name = "quantization",
    srcs = ["quantization.py"],
    srcs_version = "PY3",
    deps = [
        ":compute_cost_utils",
        ":fp_cast",
        ":get_bounds",
        ":primitives",
        ":shape_utils",
        ":utils",
        "//third_party/google_research/google_research/aqt/jax/flax:struct",
        "//third_party/py/absl/flags",
        "//third_party/py/flax:core",
        "//third_party/py/jax",
    ],
)

pytype_strict_library(
    name = "flax_layers",
    srcs = ["flax_layers.py"],
    srcs_version = "PY3",
    deps = [
        ":compute_cost_utils",
        ":get_bounds",
        ":quant_config",
        ":quantization",
        ":shape_utils",
        ":stats_tag",
        ":utils",
        "//third_party/google_research/google_research/aqt/jax/flax:struct",
        "//third_party/py/absl/flags",
        "//third_party/py/flax:core",
        "//third_party/py/jax",
    ],
)

pytype_strict_library(
    name = "flax_attention",
    srcs = ["flax_attention.py"],
    srcs_version = "PY3",
    deps = [
        ":flax_layers",
        ":get_bounds",
        ":quant_config",
        ":quantization",
        ":shape_utils",
        ":stats_tag",
        "//third_party/google_research/google_research/aqt/jax/flax:struct",
        "//third_party/py/flax:core",
        "//third_party/py/jax",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "get_bounds",
    srcs = ["get_bounds.py"],
    srcs_version = "PY3",
    deps = [
        ":quant_config",
        ":shape_utils",
        ":stats",
        "//third_party/google_research/google_research/aqt/jax/flax:struct",
        "//third_party/py/flax:core",
        "//third_party/py/jax",
    ],
)

pytype_strict_library(
    name = "stats",
    srcs = ["stats.py"],
    srcs_version = "PY3",
    deps = [
        ":shape_utils",
        "//third_party/google_research/google_research/aqt/jax/flax:struct",
        "//third_party/py/flax:core",
        "//third_party/py/jax",
    ],
)

pytype_strict_library(
    name = "stats_tag",
    srcs = ["stats_tag.py"],
    srcs_version = "PY3",
    deps = [
        ":shape_utils",
        ":stats",
        ":utils",
        "//third_party/py/flax:core",
        "//third_party/py/jax",
    ],
)

pytype_strict_library(
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY3",
)

pytype_strict_library(
    name = "quant_config",
    srcs = ["quant_config.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/google_research/google_research/aqt/jax/flax:struct",
        "//third_party/py/flax:core",
    ],
)

pytype_strict_library(
    name = "shape_utils",
    srcs = ["shape_utils.py"],
    srcs_version = "PY3",
)

pytype_strict_library(
    name = "train_utils",
    srcs = ["train_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/google_research/google_research/aqt/jax:quant_config",
    ],
)

pytype_strict_library(
    name = "hlo_utils",
    srcs = ["hlo_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//pyglib:gfile",
        "//third_party/py/flax:core",
        "//third_party/py/jax",
        "//third_party/tensorflow/compiler/xla/service:hlo_pb2",
    ],
)

pytype_strict_library(
    name = "compute_cost_utils",
    srcs = ["compute_cost_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":hlo_utils",
        "//third_party/py/absl/flags",
        "//third_party/py/jax",
        "//third_party/py/numpy",
        "//third_party/tensorflow/compiler/xla/service:hlo_pb2",
    ],
)

py_strict_test(
    name = "train_utils_test",
    srcs = ["train_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":train_utils",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "compute_cost_utils_test",
    srcs = ["compute_cost_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":compute_cost_utils",
        ":flax_layers",
        ":get_bounds",
        ":hlo_utils",
        ":quant_config",
        ":quantization",
        "//third_party/py/absl/flags",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/flax:core",
        "//third_party/py/jax",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "hlo_utils_test",
    srcs = ["hlo_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":hlo_utils",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/flax:core",
        "//third_party/py/jax",
    ],
)
