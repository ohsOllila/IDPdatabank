# SASGP

Gaussian-process tooling for membrane form factors and real-space electron-density profiles, using a cosine transform.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
# Optional NUTS inference and notebooks:
pip install -e '.[nuts,notebook]'
```

Python 3.9+ is supported; `tomli` supplies TOML parsing before Python 3.11.
The `sasgp` package exports the existing GP and transform API. `import gptransform`
remains supported for existing scripts and saved models. The former README's
`fit_gp` and `preprocess_xff` helpers are not implemented in this checkout.

## Layout and usage

- `gptransform.py`: GP, transforms, fitting metrics, Laplace and NUTS inference.
- `sasgp/__init__.py`: public imports.
- `membrane/run.py`: batch workflow over the supplied NumPy datasets.
- `membrane/config.toml`: preprocessing, model bounds, inference and output settings.
- `tests/`: regression checks.
- `IDPs/`: protein SAXS workflow development, before databank integration.

```bash
python membrane/run.py --config membrane/config.toml --index 238
python membrane/run.py --config membrane/config.toml --index 238 --inference nuts
python -m unittest discover -s tests -v
```

Example of the installed transform API:

```python
import torch
from sasgp import ed2ff

r = torch.linspace(-5, 5, 101, dtype=torch.float64)
q = torch.linspace(0, 2, 50, dtype=torch.float64)
density = torch.exp(-r**2)
form_factor = ed2ff(r, density, q)
```

## Model conventions

The cosine mean uses `-A*cos(pi*r/B)` within `abs(r/B) <= 1.5`, so `B` is a
length scale. MAP, Laplace and NUTS share this mean. NUTS uses trapezoidal
endpoint weights and respects `prior_fn`: `uniform` means uniform within the
physical parameter bounds; `gaussian` means Normal(0, 3) on raw parameters.
NUTS retains its numerical diagonal jitter; exact floating-point agreement
with the main GP's positive-definite projection is not guaranteed.
Results from older NUTS runs may differ after these corrections.

## Preprocessing and IDP integration

`data.start_index`, `cutoff`, `q_scale`, `recover_sign`, `r_min`, `r_max`,
and `slicing` control preprocessing. Defaults preserve the membrane workflow:
drop the first 76 points, multiply q by 10, recover the form-factor sign, and
use a real-space grid from -5 to 5. Quadratic noise now respects the configured
`sigma_n_base` bounds instead of silently overriding them.

The runner still consumes membrane form-factor and reference-density NumPy
arrays. It is not an IDP SAXS intensity loader. Do not apply its sign recovery
or interpret its electron-density output as a protein pair-distance
distribution without an appropriate forward model. IDP integration still
requires selecting that model and establishing the input units. Experimental
uncertainty support is not added by this change.

## Citation information
If you use this repository, please cite the original GP implementation here:

 Sullivan, H. W., Cervenka, M., Shanks, B. L. & Hoepfner, M. P. Physics-Informed Gaussian Process Inference of Liquid Structure from Scattering Data. J. Phys. Chem. B 129, 11802–11815 (2025).

