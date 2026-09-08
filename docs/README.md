# Documentation

The documentation is built with [Sphinx](https://www.sphinx-doc.org/) and
published on Read the Docs (see `.readthedocs.yaml` in the repository root).

To build it locally, run from the repository root:

```bash
tox -e docs
```

The HTML output lands in `docs/build/html`. Open `docs/build/html/index.html`
in a browser.

The `docs` tox environment first runs `docs/src/run_apidoc.py`, which calls
`sphinx-apidoc` to generate one `.rst` file per module of the `fairmd.idp`
package into `docs/src/auto_gen/` (git-ignored). Then `sphinx-build` renders
everything under `docs/src/`.

Since importing `fairmd.idp` requires the `Data` folder, the tox environment
sets `NMLDB_ROOT_PATH` to the repository root. If you run Sphinx by hand, do
the same:

```bash
export NMLDB_ROOT_PATH=$(pwd)
python docs/src/run_apidoc.py
sphinx-build -E --builder html docs/src docs/build/html
```

### Custom module rst files

If you want to hand-write the page for an individual module, look at the
name of the file produced by `run_apidoc.py` inside `docs/src/auto_gen`, for
instance `fairmd.idp.core.rst`, and put a file with the same name in that
folder. `sphinx-apidoc` does not overwrite existing files.

### Custom apidoc and sphinx templates

The templates in `docs/src/_templates` are taken over from
[FAIRMD_lipids](https://github.com/NMRLipids/FAIRMD_lipids). The footer adds
links to the repository and the license; these links are configured through
the `html_context` dictionary in `docs/src/conf.py`. The apidoc package
template removes the distinction between namespace packages and normal
packages.
