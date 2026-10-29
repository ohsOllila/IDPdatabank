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
- `gp_quality_draws.py`: CRPS against the full equal-weight Gaussian mixture of
  per-draw posteriors instead of the single-Gaussian summary. The pairwise
  mixture term is evaluated in closed form, so this is the exact expectation of
  draw-based CRPS; components are reconstructed from saved hyperparameter
  chains (validated against the saved mean and latent SD) and cached per
  experiment.
- `plot_crps_gaussian_vs_draws.py`: per-experiment comparison of single-Gaussian
  and draw-mixture CRPS/SD, styled like `plot_crps_bift_comparison.py`. Across
  all five datasets the two scores agree at Spearman rho >= 0.9999, with median
  relative score changes below 6% and no rank swaps among 209 comparisons, so
  the Gaussian summary is empirically rank-preserving here.
- `plot_method_disagreement.py`: for the largest GP-vs-BIFT rank disagreements,
  standardizes the same fitted simulation residual by the GP predictive SD and
  by the reported experimental sigma, localizing whether the disagreement comes
  from the uncertainty model (sigma ratio panel) or from low-q versus high-q
  residual structure. Writes per-case figures and `disagreement_anatomy.csv`.
- `run_synthetic_truth_study.py`: synthetic ground-truth robustness test. A
  two-Gaussian coil P(r) matched to SASDQK7's fitted Rg, intensity, and
  noise level is perturbed 19 physically graded ways (size scaling, shoulder
  weight, tail length, compact bumps, broadening, tail tilt, large-distance
  truncation), giving 20 simulations with an exact noise-free discrepancy
  ranking. Four error-bar scenarios (calibrated, low-q overestimated bars,
  counting-statistics relative bars, low-q-corrupted noise with constant bars)
  generate synthetic experiments; the GP is refit per scenario and its ranking
  compared with BIFT-convention reduced chi-squared. The BIFT Fortran binary is
  not recompiled: its global error-correction factor is rank-invariant, and its
  rebinned grid is the only unmodeled difference.
  Headline result: with calibrated constant error bars both methods rank well
  (GP Spearman rho 0.989 vs truth, chi_r^2 0.983), but relative
  counting-statistics error bars against constant noise destroy chi_r^2
  (rho 0.260, top-5 overlap 0.0, chi_r^2 ~ 1e16 even for the exact truth),
  and low-q-overestimated bars compress its discrimination
  (rho 0.850, top-5 0.4). The GP is invariant to reported bars by construction
  (identical rho 0.989 in all bar scenarios) but degrades honestly when its
  constant-noise assumption is violated (rho 0.920 in the noisy-low-q scenario,
  where chi_r^2 retains 0.982 because its constant bars match the constant
  noise contribution once the inflated low-q points are absorbed by the offset).
- `test_synthetic_truth.py`: tests for the truth model, chi-squared convention,
  and exact discrepancy.
- `test_saxs_gp.py`: mathematical and scoring tests.

```bash
MPLCONFIGDIR=IDPs/.mplconfig IDPs/.venv/bin/python \
  -m unittest discover -s IDPs -p 'test_*.py'
```

A fit is flagged unless two chains give split R-hat below 1.01, effective sample
size above 100 for every parameter, and zero divergences.
