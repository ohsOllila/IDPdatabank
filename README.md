# FAIRMD protein databank
This is the repository for the development of the FAIRMD protein databank, which is an overlay databank containing molecular dynamics (MD) simulations of proteins with programmatic access and quality evaluations against experiments. This is an extension of the [NMRlipids databank](https://doi.org/10.26434/chemrxiv-2023-jrpwm) (now [FAIRMD lipids databank](https://github.com/NMRLipids/FAIRMD_lipids/)).

README.yaml files contain all the essential information for each simulation, including the permanent location of each simulation file, enabling the data upcycling and reuse. The README.yaml files are located in [Data/simulations](https://github.com/NMRLipids/Databank/tree/main/Data/Simulations) folder under subfolders named based on file hash identities. After installation of the databank, simulations can be programmatically accessed and analyzed. For example, see [calcProperties.py](https://github.com/ohsOllila/IDPdatabank/blob/master/Scripts/AnalyzeDatabank/calcProperties.py) and [plotQuality.ipynb](https://github.com/ohsOllila/IDPdatabank/blob/master/Scripts/AnalyzeDatabank/plotQuality.ipynb) and other analysis codes in this repository.

[CalcProperties.py](https://github.com/ohsOllila/IDPdatabank/blob/master/Scripts/AnalyzeDatabank/calcProperties.py) analyzes automatically the contact, distance and backbone correlation maps, radius of gyrations, dynamic landscapes, spin relaxation times, chemical shifts, SAXS intensities, secondary structures and folding state of proteins from all simulations in the databank, and stores the results in the same folders with the README.yaml files.

Simulation entries can be added using [AddData.py](https://github.com/ohsOllila/IDPdatabank/blob/master/Scripts/BuildDatabank/AddData.py) function and info.yaml files, similarly to the NMRlipids databank (see [FAIRMD lipids documentation](https://databank.readthedocs.io/stable/contrib/addingSimulation.html#addsimulation) but note that the upload portal is not yet implemented for proteins).

Experimental data from BMRB can be automatically fetched with [create_experimental_data_file.py](https://github.com/ohsOllila/IDPdatabank/blob/master/Scripts/BuildDatabank/create_experimental_data_file.py).

The databank is being developed in [FAIRMD - Disorder to Order: Streamlining Biomolecule Simulation Re-Use with FAIR NMRlipids database project](https://www.oscars-project.eu/projects/fairmd-disorder-order-streamlining-biomolecule-simulation-re-use-fair-nmrlipids-database) [funded by Open Science Clusters' Action for Research and Society (OSCARS)](https://www.oscars-project.eu/). 


## Publication
Manuscript describing the FAIRMD protein databank is being prepared. People contributing this repository will be invited authors according to the [authorship document](https://github.com/ohsOllila/IDPdatabank/blob/master/AUTHORSHIP.md) adapted from the [NMRlipids project](https://nmrlipids.blogspot.com/).


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
