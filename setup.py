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


"""setup.py for AQT."""

import os
from setuptools import find_packages
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))
try:
  README = open(os.path.join(here, "README.md"), encoding="utf-8").read()
except IOError:
  README = ""

jax_legacy_install_requires = [
    "numpy>=1.16.4",
    "jax>=0.3.4",
    "flax>=0.4.1",
    "tensorflow>=2.3.1",
    "tensorflow-datasets>=4.5.2",
    "dacite>=1.6.0",
    "ml_collections>=0.1.1",
]

setup(
    name="aqtp",
    version="0.1.0",
    description="AQT: Accurate Quantized Training",
    long_description="\n\n".join([README]),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="",
    author="Cerebra Catalyst team",
    author_email="cerebra-catalyst-team@google.com",
    url="https://github.com/google/aqt",
    packages=find_packages(),
    python_requires=">=3.10",
    zip_safe=False,
    extras_require={
        "jax_legacy": jax_legacy_install_requires,
    },
)
