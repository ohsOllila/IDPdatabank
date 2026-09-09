.. _api:

API reference
=============

The ``fairmd.idp`` package is organised as follows.

:mod:`fairmd.idp`
   Package level constants: the data paths ``NMLDB_ROOT_PATH``,
   ``NMLDB_DATA_PATH``, ``NMLDB_SIMU_PATH``, ``NMLDB_MOL_PATH``,
   ``NMLDB_EXP_PATH`` and the return codes ``RCODE_SKIPPED``,
   ``RCODE_COMPUTED``, ``RCODE_ERROR``.

:mod:`fairmd.idp.core`
   Data model of the databank (:class:`~fairmd.idp.core.System`,
   :class:`~fairmd.idp.core.SystemsCollection`) and
   :func:`~fairmd.idp.core.initialize_databank`.

:mod:`fairmd.idp.databankLibrary`
   Helpers to access simulation files, e.g. building an MDAnalysis universe
   from a system, mapping between simulation specific and universal atom
   names, and validating ``info.yaml`` files.

:mod:`fairmd.idp.databankio`
   Downloading and resolving trajectory files from their DOI.

:mod:`fairmd.idp.protein_functions`
   IDP specific analyses and quality evaluation against SAXS, chemical shift
   and spin relaxation experiments, including the retrieval of experimental
   data from the BMRB.

:mod:`fairmd.idp.settings`
   Molecule, mapping and simulation engine definitions.

The remaining modules (:mod:`fairmd.idp.analyze`, :mod:`fairmd.idp.quality`,
:mod:`fairmd.idp.form_factor`, ...) are inherited from the lipid databank.

.. toctree::
   :maxdepth: 2

   auto_gen/fairmd.idp
