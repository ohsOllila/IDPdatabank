This is preliminary repository for databank of IDP simulations developed in the FAIRMD project.

This works in the same way a the [NMRlipids databank](https://github.com/NMRLipids/Databank)

AddData creates README files where proteins are defined using FASTA sequence

CalcProperties currently calculates contact, distance and backbone correlation maps, radius of gyration distributions, dynamic landscapes and spin relaxation times.


## Installation

The library is distributed as the `fairmd.idp` package (distribution name `fairmd-idp`),
following the layout of [FAIRMD_lipids](https://github.com/NMRLipids/FAIRMD_lipids):

```bash
pip install git+https://github.com/ohsOllila/IDPdatabank.git
```

or, from a clone of this repository, as an editable install:

```bash
pip install -e .
```

Alternatively, create a conda environment with all dependencies from
conda-forge and install the package into it:

```bash
mamba env create -f environment.yml
mamba activate fairmd-idp
pip install --no-deps -e .
```

The scripts in `Scripts/` import the library as `fairmd.idp`, e.g.

```python
from fairmd.idp.core import *
from fairmd.idp.protein_functions import *
```

When the package is not installed from a clone of this repository, set
`NMLDB_ROOT_PATH` to the cloned repository folder (or `NMLDB_DATA_PATH` to its
`Data` folder) so that the data can be found.

## Development

Linting, tests, package build and documentation are run with [tox](https://tox.wiki/):

```bash
pip install tox
tox -e lint    # ruff + sphinx-lint
tox -e tests   # pytest
tox -e build   # build sdist/wheel and check the manifest
tox -e docs    # build the documentation into docs/build/html
```

The documentation is published at <https://ohsollila.github.io/IDPdatabank/>
on every push to `master`. See `docs/src/development.rst` and `docs/README.md`
for details.
