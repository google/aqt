load("//devtools/python/blaze:strict.bzl", "py_strict_test")
load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

pytype_strict_library(
    name = "config_schema",
    srcs = ["config_schema.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/google_research/google_research/aqt/utils:config_schema_utils",
        "//third_party/py/ml_collections",
    ],
)

py_strict_test(
    name = "config_schema_test",
    srcs = ["config_schema_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":config_schema",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
    ],
)
