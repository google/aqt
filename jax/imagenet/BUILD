load("//devtools/python/blaze:pytype.bzl", "pytype_strict_binary", "pytype_strict_library")
load("//devtools/python/blaze:strict.bzl", "py_strict_test")

# BEGIN GOOGLE-INTERNAL
load("//learning/deepmind/python/fragmented_python:build_defs.bzl", "fragmented_py_binary_mpms")
# END GOOGLE-INTERNAL

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

pytype_strict_library(
    name = "all_configs",
    srcs = glob([
        "configs/*.py",
        "configs/**/*.py",
        "configs/**/**/*.py",
    ]),
    deps = [
        "//third_party/google_research/google_research/aqt/jax/imagenet/configs_script:config_schema",
        "//third_party/py/ml_collections",
    ],
)

pytype_strict_binary(
    name = "check_config_util",
    srcs = ["check_config_util.py"],
    data = glob([
        "configs/*.py",
        "configs/**/*.py",
        "configs/**/**/*.py",
    ]),
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":all_configs",  # buildcleaner: keep
        ":hparams_config",
        ":models",
        "//third_party/google_research/google_research/aqt/utils:hparams_utils",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/ml_collections/config_flags",
    ],
)

pytype_strict_library(
    name = "train_utils",
    srcs = [
        "train_utils.py",
    ],
    deps = [
        ":input_pipeline",
        ":models",
        "//third_party/google_research/google_research/aqt/jax:quant_config",
        "//third_party/py/flax:core",
        "//third_party/py/flax/training",
        "//third_party/py/jax",
        "//third_party/sentencepiece/src/python:sentencepiece_trainer",
    ],
)

pytype_strict_library(
    name = "input_pipeline",
    srcs = [
        "input_pipeline.py",
    ],
    deps = [
        "//third_party/py/jax",
        "//third_party/py/tensorflow",
        "//third_party/py/tensorflow_datasets",
        "//third_party/sentencepiece/src/python:sentencepiece_trainer",
    ],
)

pytype_strict_library(
    name = "models",
    srcs = [
        "models.py",
        # BEGIN GOOGLE-INTERNAL
        "google/pokebnn.py",
        # END GOOGLE-INTERNAL
    ],
    deps = [
        "//third_party/google_research/google_research/aqt/jax:flax_layers",
        "//third_party/google_research/google_research/aqt/jax:quant_config",
        "//third_party/google_research/google_research/aqt/jax/flax:struct",
        "//third_party/py/flax:core",
        "//third_party/py/jax",
    ],
)

pytype_strict_library(
    name = "hparams_config",
    srcs = ["hparams_config.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/google_research/google_research/aqt/jax/flax:struct",
        "//third_party/google_research/google_research/aqt/utils:hparams_utils",
    ],
)

py_strict_test(
    name = "models_test",
    timeout = "long",
    srcs = [
        "models_test.py",
    ],
    data = glob(["configs/**"]),
    python_version = "PY3",
    shard_count = 4,
    srcs_version = "PY3",
    deps = [
        ":all_configs",
        ":hparams_config",
        ":models",
        ":train_utils",
        "//third_party/google_research/google_research/aqt/jax:flax_layers",
        "//third_party/google_research/google_research/aqt/jax:hlo_utils",
        "//third_party/google_research/google_research/aqt/utils:hparams_utils",
        "//third_party/py/absl/flags",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/jax",
    ],
)

py_strict_test(
    name = "resnet_cost_test",
    timeout = "eternal",
    srcs = [
        "resnet_cost_test.py",
    ],
    data = glob(["configs/**"]),
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":all_configs",
        ":hparams_config",
        ":models",
        ":train_utils",
        "//third_party/google_research/google_research/aqt/jax:compute_cost_utils",
        "//third_party/google_research/google_research/aqt/jax:flax_layers",
        "//third_party/google_research/google_research/aqt/jax:hlo_utils",
        "//third_party/google_research/google_research/aqt/utils:hparams_utils",
        "//third_party/py/absl/flags",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/jax",
    ],
)

py_strict_test(
    name = "train_test",
    timeout = "long",
    srcs = [
        "train_test.py",
    ],
    data = glob(["configs/**"]),
    python_version = "PY3",
    shard_count = 4,
    srcs_version = "PY3",
    tags = ["optimized_only"],
    deps = [
        ":all_configs",
        ":hparams_config",
        ":models",
        ":train_utils",
        "//third_party/google_research/google_research/aqt/utils:hparams_utils",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/jax",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "configs_test",
    srcs = [
        "configs_test.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":all_configs",  # buildcleaner: keep
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
    ],
)

pytype_strict_binary(
    name = "imagenet",
    srcs = ["train.py"],
    main = "train.py",
    python_version = "PY3",
    deps = [":imagenet_lib"],
)

pytype_strict_library(
    name = "imagenet_lib",
    srcs = [
        "train.py",
    ],
    data = glob(["configs/**"]),
    srcs_version = "PY3",
    deps = [
        ":all_configs",  # buildcleaner: keep
        ":hparams_config",
        ":input_pipeline",
        ":models",
        ":train_utils",

        # BEGIN GOOGLE-INTERNAL
        "//learning/brain/research/jax:gpu_support",  # buildcleaner: keep
        "//learning/brain/research/jax:tpu_support",  # buildcleaner: keep
        "//learning/deepmind/xmanager2/client/google",  # buildcleaner: keep
        "//perftools/accelerators/xprof/xprofilez/cpu:python_tracer",  # buildcleaner: keep
        "//pyglib:gfile",
        "//pyglib/contrib/gpathlib:gpath",
        # END GOOGLE-INTERNAL
        "//third_party/google_research/google_research/aqt/jax:compute_cost_utils",
        "//third_party/google_research/google_research/aqt/jax:hlo_utils",
        "//third_party/google_research/google_research/aqt/jax:train_utils",
        "//third_party/google_research/google_research/aqt/utils:hparams_utils",
        "//third_party/google_research/google_research/aqt/utils:report_utils",
        "//third_party/google_research/google_research/aqt/utils:summary_utils",
        "//third_party/py/vision_transformer/vit_jax",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/absl/logging",
        "//third_party/py/flax",
        "//third_party/py/flax:core",
        "//third_party/py/jax",
        "//third_party/py/ml_collections/config_flags",
        "//third_party/py/tensorflow",
        "//third_party/py/numpy",
    ],
)

# BEGIN GOOGLE-INTERNAL
filegroup(
    name = "xmanager",
    srcs = ["google/xm_launch.py"],  # build_cleaner exemption
    tags = ["ignore_srcs"],
    visibility = ["//visibility:private"],
)

fragmented_py_binary_mpms(
    name = "imagenet_fragmented_mpms",
    binary = ":imagenet",
    fragments = [
        "//third_party",
        "//third_party/py/flax",
    ],
    package_name_prefix = "test/third_party/google_research/google_research/aqt/jax/imagenet",
)
# END GOOGLE-INTERNAL
