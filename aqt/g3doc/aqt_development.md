# For AQT Developers

This is a documentation for AQT developers.

## Instruction for publishing pip package

First update version number in `__init__.py`` and commit. Then run these
commands:

```
git clone git@github.com:google/aqt.git
cd aqt
rm -fr /tmp/aqt-venv
virtualenv /tmp/aqt-venv
source /tmp/aqt-venv/bin/activate
flit build
# make sure the version number on whl file is correct.
flit publish
```

## Run flaky test

AQT leverages the exactness of training loss in the end-to-end model test to
justify if a commit is a pure refactoring. The assertion on loss is subject to
two limitations:

1. Different CPU models, e.g., skylake or milan, are not bit exact and sometimes
produces different losses under the same training setting. It is important to
run the test multiple times to enumerate the CPU backend in the TAP server.

2. Compiler changes will not always be bit exact. The training loss will change
from time to time as the compiler evolves. It is important to update the loss.

When submitting code to AQT that is not a pure refactoring,
use the following command to run the flax_e2e_model_test on CPU for 50 times.
Usually it is sufficient to capture the flakiness.

```
blaze  test --test_filter=MnistTest --test_output=errors //third_party/py/aqt/jax/v2/examples:flax_e2e_model_test --runs_per_test_detects_flakes --runs_per_test=50
```
