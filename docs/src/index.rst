.. _index:

FAIRMD IDP Databank
===================

FAIRMD IDP is a preliminary databank of molecular dynamics (MD) simulations of
intrinsically disordered proteins (IDPs) developed in the FAIRMD project. It
works in the same way as the `FAIRMD Lipids databank
<https://github.com/NMRLipids/FAIRMD_lipids>`_ (formerly NMRlipids Databank):
simulations are described by metadata files, analysed with a common set of
scripts, and compared against experimental data such as small-angle X-ray
scattering (SAXS), NMR chemical shifts and NMR spin relaxation times.

FAIRMD IDP is an overlay databank
---------------------------------

Each simulation entry contains a ``README.yaml`` file that stores all the
essential information for reusing the data, such as the permanent location of
each trajectory file (a DOI), the force field, the protein sequence and the
simulation conditions. The raw trajectories stay in distributed locations
outside the databank, typically `Zenodo <https://zenodo.org/>`_. The properties
computed from the trajectories are stored next to the ``README.yaml`` file.
The organisation of the ``Data`` folder is described in :ref:`dbstructure`.

Python API
----------

The library that backs the scripts is distributed as the ``fairmd.idp``
package. It provides the initialisation of the databank
(:mod:`fairmd.idp.core`), general helpers to access simulation files
(:mod:`fairmd.idp.databankLibrary`, :mod:`fairmd.idp.databankio`) and the
protein specific analyses and quality evaluations
(:mod:`fairmd.idp.protein_functions`). See :ref:`gettingstarted` for a minimal
example and :ref:`api` for the full reference.

Adding simulations and running analyses
---------------------------------------

Simulations are added with ``AddData.py`` from an ``info.yaml`` file, and the
properties are computed with ``calcProperties.py``. The available scripts are
described in :ref:`dbprograms`.

Installation
------------

The package is installed with ``pip`` straight from GitHub:

.. code-block:: bash

   pip install git+https://github.com/ohsOllila/IDPdatabank.git

or, from a clone of the repository, as an editable install:

.. code-block:: bash

   git clone https://github.com/ohsOllila/IDPdatabank.git
   cd IDPdatabank
   pip install -e .

The scripts and the library need access to the ``Data`` folder of the cloned
repository. Set ``NMLDB_ROOT_PATH`` to the cloned folder when the package is
not installed in editable mode from that clone. See :ref:`gettingstarted`.

.. toctree::
   :maxdepth: 2
   :caption: Python Interface

   gettingstarted
   dbprograms
   api

.. toctree::
   :maxdepth: 2
   :caption: IDP Databank

   dbstructure

.. toctree::
   :maxdepth: 1
   :caption: Development

   development

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
