.. _dbstructure:

Databank structure
==================

All data of the databank lives in the ``Data`` folder of the repository:

.. code-block:: text

   Data/
   ├── info_files/      input files for AddData.py, one folder per simulation
   ├── Simulations/     one folder per simulation, addressed by file hashes
   ├── Experiments/     experimental data, one folder per experiment type
   └── Molecules/       molecule definitions and mapping files

.. _info_yaml:

Input files (``Data/info_files``)
---------------------------------

Each simulation is added with an ``info.yaml`` file that names the raw data
and describes the system. A minimal example:

.. code-block:: yaml

   DOI: 10.5281/zenodo.15393821
   SOFTWARE: gromacs
   TRJ: replica_01_AMBER03WS_md_1500ns.xtc
   TPR: replica_01_AMBER03WS_md_1500ns.tpr
   PREEQTIME: 0
   TIMELEFTOUT: 0

   COMPOSITION:
     PROTEIN:
     SOL:
       NAME: SOL
       MAPPING: mappingTIP4PSwater.yaml
     CLA:
       NAME: CL
       MAPPING: mappingCLAecc.yaml

   DIR_WRK: /tmp/tmpData/
   PUBLICATION: https://doi.org/...
   AUTHORS_CONTACT: Name
   SYSTEM: ChiZ1-64
   FF: AMBER03WS

``DOI`` points to the repository (typically Zenodo) that holds the trajectory
``TRJ`` and topology ``TPR``. ``COMPOSITION`` lists the molecules in the
system by their universal names; the ``PROTEIN`` entry is filled in by
:ref:`add_data_py` with the sequence extracted from the topology, and the
other entries refer to mapping files in ``Data/Molecules``.

Simulations (``Data/Simulations``)
----------------------------------

Simulation folders are addressed by the SHA1 hashes of the trajectory and
topology files, split into a directory tree, e.g.
``Data/Simulations/8e2/c75/8e2c75ee.../5e0ddfac.../``. Each folder contains
the ``README.yaml`` written by :ref:`add_data_py` and the properties computed
by the analysis scripts:

``README.yaml``
   The metadata of the simulation. Besides the fields copied from
   ``info.yaml`` it contains ``ID``, ``TYPEOFSYSTEM`` (``protein``),
   ``TEMPERATURE``, ``NUMBER_OF_ATOMS``, ``TRAJECTORY_SIZE``, ``TRJLENGTH``,
   ``DATEOFRUNNING``, the ``COMPOSITION`` with molecule counts and the
   ``PROTEIN`` sequence, and the ``EXPERIMENT`` block listing the matched
   ``chemical_shift``, ``saxs`` and ``spin_relaxation`` experiments together
   with their alignment score and condition matches.

``Contact_map.png``, ``Distance_map.png``
   Residue-residue contact probabilities and distances.

``Backbone_correlations.csv``, ``Backbone_correlations.png``
   Backbone N-H correlation functions.

``dynamic_landscape_Coeffs.yaml``
   Coefficients of the dynamic landscape fit.

``spin_relaxation_times.yaml``
   Computed NMR spin relaxation times.

``gyrate.xvg``
   Radius of gyration along the trajectory.

``SAXS.yaml``
   Computed SAXS profile.

``chemical_shifts_sparta.yaml``, ``chemical_shift_rmsd.yaml``, ``chemical_shift_quality.yaml``
   Computed chemical shifts and their comparison to experiments.

``secondary_structure.yaml``
   Secondary structure populations.

Experiments (``Data/Experiments``)
----------------------------------

Experiments are grouped by type, ``saxs``, ``chemical_shift`` and
``spin_relaxation``, and then by the DOI or database ID of the publication:

.. code-block:: text

   Data/Experiments/saxs/10.1038/srep30473/
   ├── fasta.yaml            protein sequence(s)
   └── saxs_metadata.yaml    sample components and conditions

``fasta.yaml`` holds the sequence used for alignment against the simulated
protein. The metadata file lists the sample components with their
concentrations (protein, buffer, salt) and the sample conditions (pH,
temperature), which ``searchDatabank.py`` uses to decide whether an
experiment matches a simulation. Chemical shift and spin relaxation
experiments are retrieved from the BMRB with
``create_experimental_data_file.py`` and additionally contain the measured
data, e.g. ``spin_relaxation_times.yaml``.

Molecules (``Data/Molecules``)
------------------------------

Molecule definitions and mapping files are organised in ``membrane``,
``protein`` and ``solution``. A mapping file connects the simulation specific
atom names of a molecule to the universal names used by the analysis code,
exactly as in the lipid databank.
