.. _dbprograms:

Databank scripts
================

The scripts live in the ``Scripts`` folder of the repository and are run from
within their own folder with a plain Python interpreter. They all import the
library as ``fairmd.idp``, so the package has to be installed first (see
:ref:`gettingstarted`).

Building the databank (``Scripts/BuildDatabank``)
-------------------------------------------------

.. _add_data_py:

AddData.py
^^^^^^^^^^

Adds a simulation into the databank based on an ``info.yaml`` file (see
:ref:`info_yaml`). The script downloads the trajectory and topology from the
DOI given in the file, checks the composition, creates the simulation folder
from the SHA1 hashes of the files and writes the ``README.yaml``.

.. code-block:: text

   AddData.py [-h] [-f FILE] [-d] [-n] [-w WORK_DIR] [-o OUTPUT_DIR]

   -h, --help                  show this help message and exit
   -f FILE, --file FILE        input config file in yaml format
   -d, --debug                 enable debug logging output
   -n, --no-cache              always redownload repository files
   -w WORK_DIR, --work-dir     custom temporary working directory
                               [not set = read from YAML]
   -o OUTPUT_DIR, --output-dir custom output directory
                               [default: NMLDB_SIMU_PATH]

Return codes: 1 for input YAML parsing errors, 2 for filesystem writing
errors, 3 for network errors.

create_IDs.py
^^^^^^^^^^^^^

Assigns a unique integer ``ID`` to every ``README.yaml`` under
``Data/Simulations`` that does not have one yet.

searchDatabank.py
^^^^^^^^^^^^^^^^^

Matches experiments with simulations. For every simulation and every
experiment folder it aligns the protein sequences, compares temperature, pH
and ionic strength with configurable thresholds
(``parameter_comparator.py``), and records the matching experiments in the
``EXPERIMENT`` block of the simulation's ``README.yaml``. Buffer compositions
and ionic strengths are handled by ``buffer_manager.py`` and
``buffer_molecule_data.py``; the classes ``Experiment`` and ``Simulation``
wrap the metadata files. See ``Scripts/BuildDatabank/README.md`` for a
description of these modules.

create_experimental_data_file.py
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fetches an NMR-STAR entry from the BMRB by its ID and creates the experiment
metadata files (sequence, conditions and data) for chemical shift and spin
relaxation experiments.

connect_to_Uniprot_and_pdb.py
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Loops over the simulations and links the protein sequences to UniProt and PDB
entries.

add_force_fields_in_READMEs.py
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Maintenance script that fills in the force field fields of existing
``README.yaml`` files.

quality_evaluation.ipynb
^^^^^^^^^^^^^^^^^^^^^^^^

Notebook to evaluate the quality of simulations against the matched
experimental data.

Analysing the databank (``Scripts/AnalyzeDatabank``)
----------------------------------------------------

calcProperties.py
^^^^^^^^^^^^^^^^^

Loops over all protein simulations and computes contact, distance and backbone
correlation maps, radius of gyration distributions, dynamic landscapes and
spin relaxation times. The results are written into each simulation folder
(see :ref:`dbstructure`). The script has no command line options and starts
computing immediately; it needs ``gmx`` in the ``PATH``.

calc_qualities.py
^^^^^^^^^^^^^^^^^

Computes the quality measures of the simulations against the matched SAXS,
chemical shift and spin relaxation experiments.

Notebooks
^^^^^^^^^

``plotting.ipynb``, ``plotQuality.ipynb``, ``stats.ipynb`` and
``chemical_shifts.ipynb`` plot the computed properties, the quality
evaluations and databank statistics.
