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
  ranking. Five scenarios (calibrated, low-q overestimated bars,
  counting-statistics relative bars, low-q-corrupted noise with constant bars,
  and a smooth low-q systematic error) generate synthetic experiments; the GP
  is refit per scenario and its ranking compared with the notebook's
  reduced-chi-squared convention (unweighted affine fit, sigma only
  normalizing residuals). The BIFT Fortran binary is not recompiled: its
  global error-correction factor is rank-invariant, and its rebinned grid is
  the only unmodeled difference.
  Because a single noise realization moves Spearman rho by +-0.2, conclusions
  require replicated realizations: run with `--replicates 8 --inference
  laplace --output-dir IDPs/outputs/synthetic_truth_replicates`. Replicated
  result (median [min,max] over 8 realizations): with calibrated or inflated
  low-q bars both methods track truth comparably (GP 0.958 [0.90,0.99] and
  0.966 [0.80,0.99]; chi_r^2 0.953 [0.88,0.99] and 0.925 [0.72,0.97]).
  Relative counting-statistics bars against constant noise are the one robust
  failure: chi_r^2 collapses to 0.468 [-0.27,0.79] (top-5 overlap 0.0) while
  the GP holds 0.929 [0.71,0.99]. Noisy low-q data degrades both comparably
  (GP 0.843 [0.57,0.98], chi_r^2 0.890 [0.46,1.00]; no reliable difference),
  and a smooth 5% low-q systematic error impairs both equally (GP 0.708
  [0.51,0.92], chi_r^2 0.687 [0.46,0.92]). Single-realization numbers
  elsewhere in outputs/synthetic_truth are NUTS fits of one draw and should
  not be quoted without these spreads.
- `run_bift_binary_study.py`: the definitive synthetic comparison, running the
  REAL BIFT binary (compiled from GenApp-BayesApp bift.f; `brew install gcc`
  then `gfortran bift.f -O2 -o bift` in Scratch/GP/SASGP/bift) on every
  replicated synthetic experiment. Scores all simulations with the notebook's
  exact compute_chi2 against BIFT's own output (rescale.dat: Shannon-rebinned
  curve, error-corrected sigma), in both of BIFT's error-correction modes:
  constant factor (the notebook's default) and non-constant q-/intensity-
  dependent correction. GP scores reuse the cached per-realization fits;
  nothing GP-related is recomputed. Reported sigma is floored at 1 a.u. for
  the counting-statistics scenario because BIFT's reader rejects sigma <= 0
  (itself notable: proportional error bars on a decaying curve crash the
  mainstream pipeline). Over 8 realizations (median [min,max] Spearman rho vs
  the exact ranking; GP on identical realizations):
  calibrated: BIFT 0.963 [0.90,0.99], GP 0.958 [0.90,0.99] -- equivalent.
  q-shaped miscalibrated bars (counting statistics): BIFT 0.641 [0.40,0.85],
  non-constant mode 0.554 -- BIFT's Shannon rebinning softens but does not
  fix it, and its own q-dependent error correction makes ranking WORSE --
  while the GP holds 0.929 [0.71,0.99]. This is the single reproducible
  advantage of the GP. Low-q inflated bars (0.922/0.941 vs 0.966), noisy
  low-q data (0.822/0.770 vs 0.843), and smooth systematic error
  (0.668/0.672 vs 0.708) leave the methods statistically indistinguishable.
- `plot_bift_binary_comparison.py`: one figure per scenario from
  `bift_scores.csv`: score-vs-truth scatter with min-max bars over the 8
  realizations for the GP and both BIFT correction modes, a median-score
  heatmap sorted by the exact ranking, and a per-realization GP-vs-BIFT rank
  scatter annotated with the summary rhos. Written to
  `outputs/synthetic_truth_bift_binary/plots/`.
- `test_synthetic_truth.py`: tests for the truth model, chi-squared convention,
  and exact discrepancy.
- `gp_quality_function.py` has been promoted into the main codebase and
  removed from Scratch: the fitting code now lives in
  `src/fairmd/idp/saxs_gp_fit.py` (run databank-wide by
  `Scripts/AnalyzeDatabank/calcSAXSGPFits.py`), the scoring code and the
  single `evaluate_SAXS_quality_GP(system, databankPath)` function live in
  `src/fairmd/idp/saxs_gp_quality.py` (called from `calcProperties.py`), and
  each experiment's portable fit is stored beside it as
  `Data/Experiments/saxs/<experiment>/gp_fit.dat`. Scoring needs only
  numpy/scipy/yaml; torch/pyro are required only for fitting. Tests are in
  `tests/test_saxs_gp_quality.py`. `run_gp_quality.py` now delegates its
  gp_fit.dat export to `fairmd.idp.saxs_gp_fit.export_gp_fit`.
- `toy_problem_summary.md`: collaborator-facing summary of the synthetic
  ground-truth study and its replicated results (the definitive numbers are
  those in this file; single-realization CSVs under ignored `outputs/` are
  working artifacts).
- `test_saxs_gp.py`: mathematical and scoring tests.

```bash
MPLCONFIGDIR=IDPs/.mplconfig IDPs/.venv/bin/python \
  -m unittest discover -s IDPs -p 'test_*.py'
```

A fit is flagged unless two chains give split R-hat below 1.01, effective sample
size above 100 for every parameter, and zero divergences.
