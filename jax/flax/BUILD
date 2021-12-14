load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

pytype_strict_library(
    name = "struct",
    srcs = ["struct.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/flax:core",
        "//third_party/py/jax",
    ],
)
