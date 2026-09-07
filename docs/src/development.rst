.. _development:

Development
===========

Development tasks are automated with `tox <https://tox.wiki/>`_, following
the setup of `FAIRMD_lipids <https://github.com/NMRLipids/FAIRMD_lipids>`_.
Install tox once (``pip install tox``) and run from the repository root:

.. code-block:: bash

   tox              # default: lint, build, tests
   tox -e tests     # run the test-suite with pytest
   tox -e lint      # ruff format check, ruff check, sphinx-lint
   tox -e format    # apply ruff formatting and auto-fixes
   tox -e build     # build sdist/wheel, twine check, check-manifest
   tox -e docs      # build the HTML documentation

Arguments after ``--`` are passed on, e.g. ``tox -e tests -- -k buffer``.

Environments
------------

``tests``
   Installs the package in editable mode with ``pytest`` and ``pytest-cov``
   and runs the tests in ``tests/``. ``NMLDB_ROOT_PATH`` is set to the
   repository root so that ``fairmd.idp`` can be imported.

``lint`` and ``format``
   `ruff <https://docs.astral.sh/ruff/>`_ is configured in ``pyproject.toml``
   (line length 120, rule sets ``E``, ``F``, ``B``, ``I``). The linted folders
   are ``src/``, ``Scripts/``, ``tests/`` and ``docs/src/``. ``sphinx-lint``
   checks the reStructuredText sources. The same checks can run before every
   commit through `pre-commit <https://pre-commit.com/>`_
   (``pre-commit install``).

``build``
   Builds the distribution and verifies that the sdist contains exactly the
   files tracked in git that are not excluded in ``MANIFEST.in``.

``docs``
   Runs ``docs/src/run_apidoc.py`` to generate the API pages and then
   ``sphinx-build``. The output is in ``docs/build/html``. See
   ``docs/README.md`` for details. Read the Docs builds the same
   configuration from ``.readthedocs.yaml``.

Continuous integration
----------------------

The GitHub workflows in ``.github/workflows`` run ``tox -e lint``,
``tox -e build``, ``tox -e tests`` and ``tox -e docs`` on pull requests.
