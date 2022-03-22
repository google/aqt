# Accurate Quantized Training

This directory contains libraries for running and analyzing
neural network quantization experiments in JAX and flax.

Summary about this work is presented at paper [Pareto-Optimal Quantized ResNet Is Mostly 4-bit](https://arxiv.org/abs/2105.03536).
Please cite the paper in your publications if you find the source code useful for your research.

Contributors: Shivani Agrawal, Lisa Wang, Jonathan Malmaud, Lukasz Lew,
Pouya Dormiani, Phoenix Meadowlark, Oleg Rybakov.

## Installation
```
# Install SVN to only download the aqt directory of Google Research.
sudo apt install subversion

# Download this directory.
svn export https://github.com/google/aqt/jax_legacy

# Upgrade pip.
pip install --user --upgrade pip

# Install the requirements from `requirements.txt`.
pip install --user -r jax_legacy/requirements.txt

# Add jax_legacy to PYTHONPATH so that its modules can be imported anywhere.
export PYTHONPATH=/path/to/parent/dir/of/jax_legacy
```

## AQT Quantization Library

`Jax` and `Flax` quantization libraries provides `what you serve is what you train`
quantization for convolution and matmul. See [this README.md](./jax/README.md).

## Reporting Tool

After a training run has completed, the reporting tool in
`report_utils.py` allows to generate a concise experiment report with aggregated
metrics and metadata. See [this README.md](./utils/README.md).
