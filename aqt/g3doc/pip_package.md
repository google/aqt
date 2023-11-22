## Instruction for publishing pip package.

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
