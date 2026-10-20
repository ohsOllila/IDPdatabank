# IDP SAXS GP quality evaluation

Self-contained Gaussian-process workflow for comparing IDPdatabank SAXS
simulations with experimental curves.

## Method

Each unique experimental curve is fitted once with a sinc-transform GP:

`I(q) = integral P(r) sin(qr)/(qr) dr`

NUTS jointly samples four real-space kernel parameters, a constant experimental
background, and a constant Gaussian observation-noise SD. The background is
excluded from the sinc transform and P(r). The model uses q and intensity; reported experimental
error bars are not inputs.
Non-finite and non-positive intensity rows are excluded before fitting. This
removes the zero-valued terminal row in 10.1039/c9cp01768b.

For each linked simulation, its CRYSOL curve is interpolated onto the GP q grid.
A nonnegative intensity scale and constant background are fitted. Trapezoid q
weights prevent densely sampled regions from dominating.

The primary metric is pointwise Gaussian CRPS. Lower is better:

- `gp_crps`: intensity units.
- `gp_crps_normalized`: divided by mean absolute GP intensity.
- `gp_crps_sd_units`: divided by mean GP predictive SD. An exact match to a
  Gaussian predictive mean gives 0.234; offsets of one and two SD give 0.602
  and 1.453.
- `gp_z_rmse` and `gp_mean_abs_z`: standardized residual summaries.
- `gp_coverage_68` and `gp_coverage_95`: q-weighted pointwise coverage.
- `gp_relative_rmse`: discrepancy relative to GP intensity.

Coverage is diagnostic because a diffuse posterior can obtain high coverage.
The full GP covariance is not inverted. Simulation frame SD is excluded because
it is not the standard error of the trajectory mean without an effective sample
size.

## Run

From `Scratch/GP/SASGP`:

```bash
python3 -m venv IDPs/.venv
IDPs/.venv/bin/python -m pip install -r IDPs/requirements.txt
MPLCONFIGDIR=IDPs/.mplconfig OPENBLAS_NUM_THREADS=1 \
  IDPs/.venv/bin/python IDPs/run_gp_quality.py
```

The runner discovers SAXS links from simulation `README.yaml` files, fits every
unique experiment, and scores all linked `SAXS.yaml` files. Curves are cut at
q = 0.5 Å⁻¹ and thinned to at most 200 points by default because GP inference
scales cubically. Existing fits are reused unless `--refit` is passed. Use
`--scale-only` to disallow the fitted background.

Outputs under `IDPs/outputs/databank_quality/`:

- `gp_quality_scores.csv`: one row per simulation–experiment comparison.
- `<experiment>/`: GP posterior, covariance, diagnostics, and plots.
- `residuals/`: pointwise fitted curves and standardized residuals.
- `crps_quality/`: simulations ranked by absolute integrated CRPS/SD, saved as
  numeric tables and high-resolution heatmaps. Each experiment uses its own
  logarithmic color scale because instrument-specific uncertainty changes the
  practical CRPS/SD baseline; colors should only be compared within a panel.

## Files

- `saxs_gp.py`: sinc-transform GP and posterior prediction.
- `bayesian_saxs.py`: NUTS inference and mixture summaries.
- `gp_quality.py`: CRPS, coverage, residuals, and q weighting.
- `run_gp_quality.py`: databank-wide CLI.
- `plot_crps_quality.py`: per-experiment absolute CRPS heatmaps and fitted curves.
- `plot_crps_bift_comparison.py`: matched GP CRPS/BIFT reduced-chi-squared
  heatmaps, fitted curves, and rank-agreement plots. Existing BIFT results are
  read from the saved output in `Scratch/saxs_bift_comparison_SO.ipynb`; that
  table contains 30 matched simulations for each of SASDQJ7 and SASDQK7.
- `plot_crps_quality.py`: absolute CRPS/SD heatmaps paired with experimental and
  fitted simulated SAXS curves using the same quality colors.
- `test_saxs_gp.py`: mathematical and scoring tests.

```bash
MPLCONFIGDIR=IDPs/.mplconfig IDPs/.venv/bin/python \
  -m unittest discover -s IDPs -p 'test_*.py'
```

A fit is flagged unless two chains give split R-hat below 1.01, effective sample
size above 100 for every parameter, and zero divergences.
