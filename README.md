#FAIRMD protein databank
This is the repository used for the FAIRMD protein databank development. FAIRMD protein databank is an overlay databank containing molecular dynamics (MD) simulations of proteins with programmatic access and quality evaluations against experiments. This is an extension of the [NMRlipids databank](https://doi.org/10.26434/chemrxiv-2023-jrpwm) (now [FAIRMD lipids databank](https://github.com/NMRLipids/FAIRMD_lipids/)).

For each simulation, there is a README.yaml which contains all the essential information for the data upcycling and reuse, including the permanent location of each simulation file. The README.yaml files are located in [Data/simulations](https://github.com/NMRLipids/Databank/tree/main/Data/Simulations) folder under subfolders named based on file hash identities. Simulations can be automatically accessed and analyzed with after performing the installation steps as expemplified, for example, in [calcProperties.py](https://github.com/ohsOllila/IDPdatabank/blob/master/Scripts/AnalyzeDatabank/calcProperties.py) and [plotQuality.ipynb](https://github.com/ohsOllila/IDPdatabank/blob/master/Scripts/AnalyzeDatabank/plotQuality.ipynb).

CalcProperties.py currently automatically analyzes contact, distance and backbone correlation maps, radius of gyrations, dynamic landscapes, spin relaxation times, chemical shifts, SAXS intensities, secondary structures and folding state of proteins.

Simulation entries can be added with [AddData.py](https://github.com/ohsOllila/IDPdatabank/blob/master/Scripts/BuildDatabank/AddData.py) using info.yaml files, similarly to the NMRlipids databank (see [documentation](https://databank.readthedocs.io/stable/contrib/addingSimulation.html#addsimulation), upload portal is not yet implemented for proteins).

Experimental data from BMRB can be automatically fetched with [create_experimental_data_file.py](https://github.com/ohsOllila/IDPdatabank/blob/master/Scripts/BuildDatabank/create_experimental_data_file.py).

The databank is being developed in [FAIRMD - Disorder to Order: Streamlining Biomolecule Simulation Re-Use with FAIR NMRlipids database project](https://www.oscars-project.eu/projects/fairmd-disorder-order-streamlining-biomolecule-simulation-re-use-fair-nmrlipids-database) [funded by Open Science Clusters' Action for Research and Society (OSCARS)](https://www.oscars-project.eu/). 


## Publication
Manuscript describing the FAIRMD protein databank is being prepared. People contributing this repository will be invited authors according to the [authorship document]() adapted from the [NMRlipids project](https://nmrlipids.blogspot.com/).


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
