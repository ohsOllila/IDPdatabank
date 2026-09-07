.. _gettingstarted:

Getting started
===============

Installation
------------

Install the ``fairmd.idp`` package into a Python environment (Python 3.9 or
newer):

.. code-block:: bash

   pip install git+https://github.com/NMRLipids/IDPdatabank.git

For development, clone the repository and install it in editable mode instead:

.. code-block:: bash

   git clone https://github.com/NMRLipids/IDPdatabank.git
   cd IDPdatabank
   pip install -e .

Some analyses in :mod:`fairmd.idp.protein_functions` need additional tools that
are not installed automatically: `PyMOL <https://pymol.org/>`_, `MDTraj
<https://www.mdtraj.org/>`_, `MAICoS <https://maicos-devel.gitlab.io/maicos/>`_
and `GROMACS <https://www.gromacs.org/>`_ (``gmx`` in the ``PATH``).

Locating the data
-----------------

The library reads the databank from the ``Data`` folder of the repository. The
location is resolved at import time from the following environment variables:

``NMLDB_ROOT_PATH``
   Root of the cloned repository. Defaults to the repository containing the
   package when it is installed in editable mode.

``NMLDB_DATA_PATH``
   The ``Data`` folder. Defaults to ``$NMLDB_ROOT_PATH/Data``.

``NMLDB_SIMU_PATH``
   The simulations folder. Defaults to ``$NMLDB_DATA_PATH/Simulations``.

If the ``Data`` folder cannot be found, importing ``fairmd.idp`` raises a
``RuntimeError``. When the package was installed from GitHub rather than from a
clone, set the root explicitly:

.. code-block:: bash

   export NMLDB_ROOT_PATH=/path/to/IDPdatabank

Minimal example
---------------

The minimum Python code to initialise the databank is

.. code-block:: python

   from fairmd.idp.core import initialize_databank

   systems = initialize_databank()

After running this, ``systems`` is an instance of
:class:`fairmd.idp.core.SystemsCollection`, which works like a list of
:class:`fairmd.idp.core.System` objects. Each system is a dictionary-like view
of the ``README.yaml`` of one simulation plus its ``path`` inside the
simulations folder. The content of ``README.yaml`` is described in
:ref:`dbstructure`. ``systems`` can be used to loop over all simulations:

.. code-block:: python

   for system in systems:
       if system["TYPEOFSYSTEM"] != "protein":
           continue
       print(system["SYSTEM"], system["FF"], system["TEMPERATURE"])

:func:`fairmd.idp.core.print_README` prints the metadata of one simulation in
a human readable format. The scripts under ``Scripts/AnalyzeDatabank`` use the
same pattern to loop over all simulations and compute properties with the
functions from :mod:`fairmd.idp.protein_functions`, e.g. contact and distance
maps, radius of gyration, backbone correlation functions, spin relaxation
times, SAXS profiles and chemical shifts.

Where to look next
------------------

* :ref:`dbprograms` describes the scripts for building and analysing the
  databank.
* :ref:`dbstructure` describes the layout of the ``Data`` folder and the
  metadata files.
* :ref:`api` is the reference of the ``fairmd.idp`` package.
