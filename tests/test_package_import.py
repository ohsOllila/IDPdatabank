"""Smoke test: the installed ``fairmd.idp`` package and all its modules import.

This is what the conda-installability check in CI runs after creating the
environment from ``environment.yml`` and installing the package with
``pip install --no-deps``.
"""

import importlib
import os
import pkgutil
from pathlib import Path

import pytest

# fairmd.idp needs the Data folder at import time; point it to this clone
# unless the caller already configured it.
os.environ.setdefault("NMLDB_ROOT_PATH", str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="module")
def package():
    return importlib.import_module("fairmd.idp")


def test_package_imports(package):
    assert os.path.isdir(package.NMLDB_DATA_PATH)
    assert package.RCODE_COMPUTED == 1


def test_all_submodules_import(package):
    names = [m.name for m in pkgutil.walk_packages(package.__path__, package.__name__ + ".")]
    assert names, "no submodules found"
    for name in names:
        importlib.import_module(name)


def test_databank_initializes(package):
    from fairmd.idp.core import initialize_databank

    systems = initialize_databank()
    assert len(systems) > 0
