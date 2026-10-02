# IDP SAXS workflow

This directory is reserved for developing the protein SAXS workflow within
SASGP before integrating it into IDPdatabank.

The existing `membrane/` pipeline is a cosine-transform model for membrane
form factors. The IDP workflow uses a separate sinc-transform GP here,
following the membrane hyperparameter inference conventions.

The preview generates synthetic noisy simulation intensities. The GP fits only
q and intensity, inferring its own constant noise SD; experimental error bars
are not required. The preview script still loads simulation YAML data.

## Simulation preview and synthetic data

`plot_simulation.py` reads simulation 294 (silk protein, DESAMBER) from the
neighboring `../IDPdatabank` checkout. It plots the original mean SAXS intensity
on linear and logarithmic axes, with the reported ±1 SD shaded. The SD is not
assumed to be the standard error of the mean. q is in inverse angstroms and
intensity is in arbitrary units; no normalization is applied to the source data. The synthetic data are fitted below.

From the SASGP root:

```bash
python3 -m venv IDPs/.venv
IDPs/.venv/bin/python -m pip install -r IDPs/requirements.txt
MPLCONFIGDIR=IDPs/.mplconfig IDPs/.venv/bin/python IDPs/plot_simulation.py
```

The PNG, extracted CSV, and source record are saved in `IDPs/outputs/`.
Use `--source /path/to/SAXS.yaml` to inspect another simulation and
`--output-dir IDPs/outputs/another_simulation` to keep its outputs separately.

The script also treats the mean curve as a synthetic experimental reference and
adds independent zero-mean Gaussian noise at each q, with constant standard
deviation equal to the arithmetic mean of the reported SD column. `--seed 42`
(the default) makes the realization reproducible. `saxs_noise_comparison.png`
shows the original and noisy curves side by side on shared linear axes, with
the CRYSOL mean overlaid on the noisy data. Negative noisy values are retained.
`saxs_synthetic.csv` records the mean, noisy intensity, constant noise SD, and
added noise. This is an assumed synthetic noise model, not a conversion of
across-frame variability into experimental uncertainty. The synthetic data are then fitted with the SAXS GP described below.

## Gaussian-chain teaching example

```bash
MPLCONFIGDIR=IDPs/.mplconfig IDPs/.venv/bin/python IDPs/toy_gaussian_chain.py
```

This generates 3,000 independent ideal chains with 60 beads. Each step has
independent Gaussian Cartesian components with variance b²/3; b = 3.8 Å is an
illustrative RMS step length, not a fixed atomic bond length or a calibrated
protein parameter. Beads have equal point-scattering weights, with no hydration,
excluded volume, or sequence interactions. No GP is fitted.

For beads s steps apart, the distance density is
p_s(r) = 4πr² [3/(2πsb²)]^(3/2) exp[-3r²/(2sb²)].
There are N-s distinct pairs with that separation along the chain. The normalized
prior mean is their weighted mixture: P(r) = sum_s 2(N-s)/[N(N-1)] p_s(r).
It integrates to one and excludes self pairs. The full normalized intensity is
I(q)/I(0) = 1/N + (1-1/N) integral P(r) sinc(qr) dr, including self scattering.
The exact finite-chain mean-square radius is b²(N²-1)/(6N).

`outputs/toy_chain/gaussian_chain.png` shows three conformations, individual
pair-separation distributions, the analytic mixture versus sampled distances,
and their scattering curves. Coordinates, CSVs, and a numerical verification
summary are saved alongside it. Options: `--beads`, `--chains`, `--step`, `--seed`,
and `--output-dir`. The analytic mixture can serve as a future GP mean; it does
not itself specify a GP covariance or guarantee positivity of GP samples.

Background: [Yale polymer physics notes: Gaussian-chain scattering](https://www.eng.yale.edu/polymers/docs/classes/polyphys/lecture_notes/6/handout6_wsu5.html).

## SAXS FTGP fit

`plot_simulation.py` now also fits the noisy synthetic intensities, using the
independent implementation in `saxs_gp.py`. Install the updated requirements
first. Pass `--skip-gp` to generate only the original previews.

The model has zero prior mean and covariance
`K(r,r') = w(r) w(r') exp(-(r-r')²/(2 ell²))`, with
`w(r) = a (1-exp(-(r/r0)²)) exp(-(r/R)²)` for nonnegative separations.
There is no reflected covariance term. Both mean and variance vanish at the
origin and asymptotically at large separation. This first fit does not use the
optional Gaussian-chain mean or enforce positivity or unit normalization.
`P(r)` absorbs the scattering scale, with units of intensity per angstrom;
`I(q) = integral P(r) sin(qr)/(qr) dr`. No membrane sign recovery is applied.
This is an effective continuous SAXS pair distribution, not the finite-bead
self-scattering model of the teaching example.

Trapezoidal quadrature transforms the prior covariance and cross-covariance.
Only q and observed intensity enter the fit. A constant Gaussian observation
noise SD is the fifth sampled parameter, alongside the four kernel parameters.
The synthetic generating noise SD and noiseless curve are not supplied to the
fitter. Both NUTS and Laplace propagate noise/kernel posterior correlations.

Hyperparameter inference now follows the membrane workflow: Pyro NUTS (default)
or a MAP/Hessian Laplace approximation in raw logit coordinates. Physical
parameters are `lower + (upper-lower)*sigmoid(raw)`. `--prior-fn uniform`
uses a uniform physical-space prior with the sigmoid Jacobian correction;
`--prior-fn gaussian` uses independent Normal(0,3) raw priors. The SAXS model
lives entirely in IDPs; the membrane code and cosine model are unchanged.
Laplace requires a positive-definite Hessian and fails explicitly otherwise.

```bash
MPLCONFIGDIR=IDPs/.mplconfig OPENBLAS_NUM_THREADS=1 IDPs/.venv/bin/python IDPs/plot_simulation.py --inference nuts --nuts-warmup 500 --nuts-samples 500 --nuts-chains 2
```

NUTS chains run sequentially with separate seeds. Outputs include raw and
physical chains (`saxs_gp_hyperparameters.npz`), trace/histogram plots, split
R-hat, ESS, divergences and acceptance rates in the JSON. Diagnostics are
flagged unless at least two chains give R-hat < 1.01, ESS > 100 and zero
divergences. These are screening checks, not proof of convergence. Increase
warmup/draw counts for flagged runs. `--inference-seed` defaults to 43 and is
independent of the synthetic-noise seed.

Defaults are `--r-max 400 --r-points 401` (angstroms). Existing parameter
bounds are retained: scaled amplitude [1e-5/r_max,100/r_max], ell
[2*dr,r_max/2], r0 [2*dr,r_max/4], R [4*dr,r_max/4], noise SD [1e-6,1] in scaled intensity units.
Amplitude and noise SD are scaled by max(abs(observed_I)); the JSON records physical bounds. Thus
these are bounded working priors tied to the chosen grid and data scale,
not externally calibrated priors. Domain/grid changes also change these
priors; assess sensitivity before physical interpretation. The tail scale
bound makes the envelope negligible at the integration edge. r_max is an
integration limit, not a physical Dmax; the displayed pair-distance axis
remains 0–10 nm.

For each hyperparameter draw, the code computes the conditional GP mean and
covariance. Total covariance is E[conditional covariance] + Cov[conditional
mean], as in membrane prediction. Plot bands are pointwise equal-tailed
quantiles of this Gaussian mixture, not mean ± 1.96 SD. CSVs retain total SD
and add interval endpoints. `saxs_gp_covariance.npz` saves the full total
covariances and the within/between variance contributions.

The third fit panel propagates uncertainty into radius of gyration using
`Rg² = M2/(2 M0)`, with `M0 = integral P(r) dr = I(0)` and
`M2 = integral r² P(r) dr`. Expanding the sinc transform gives the Guinier
relation `I(q) ≈ I(0) exp(-q² Rg²/3)` at small q;
see https://journals.iucr.org/j/issues/2016/05/00/vg5047/.
For every hyperparameter draw, 50 joint Gaussian moment draws preserve
M0/M2 correlations before taking the ratio and square root. The pooled Rg
distribution includes conditional GP, kernel, and noise-parameter uncertainty.
The calculation uses the full integration domain; extra observation noise
is not added to latent moment draws. The CSV records the hyperparameter-draw
index, and invalid moments (M0 <= 0 or M2 < 0) are flagged with NaN Rg. Their
fraction is reported and intervals condition on valid moments. Neither
pointwise positivity nor uncertainty in model choice is included. Conditional
draws do not increase the effective number of hyperparameter samples.

Outputs: `saxs_gp_fit.png`, `saxs_gp_fit.json`, `saxs_gp_real_space.csv`,
`saxs_gp_intensity.csv`, `saxs_gp_rg_samples.csv`, and the covariance/chain
artifacts above. Existing files in the output directory are overwritten.

Run the analytic-moment, covariance-propagation, and invalid-moment checks with:

```bash
MPLCONFIGDIR=IDPs/.mplconfig IDPs/.venv/bin/python -m unittest discover -s IDPs -p 'test_*.py'
```

The `noise_sigma` JSON field is now the posterior median, with a separate
`noise_sigma_interval_95`. The fifth hyperparameter trace is the noise SD in
original intensity units. Predictive intensity variance is latent mixture
variance plus E[sigma_noise²], not the square of its posterior median. Noise
changes latent GP and Rg uncertainty through conditioning; fresh observation
noise is added only for future intensity measurements. The synthetic CSV
continues to record the known generating noise as a reference.

Direct use (q in inverse angstroms, no experimental errors required):

```python
from pathlib import Path
from saxs_gp import fit_saxs  # with IDPs on the Python import path
report = fit_saxs(q, intensity, Path('IDPs/outputs/experiment'))
```

Validation policy: use NUTS for inference validation. Do not add or run Laplace tests or Laplace validation fits unless explicitly requested by the user. The existing unit tests cover GP mathematics and the NUTS likelihood; none runs Laplace.

## First experimental fit

```bash
MPLCONFIGDIR=IDPs/.mplconfig OPENBLAS_NUM_THREADS=1 IDPs/.venv/bin/python IDPs/fit_experiment.py
```

This runs NUTS on SASDNV6, using all 290 points with q <= 0.15 Å⁻¹ for a
first lower-q fit. The native q units are assumed to be inverse angstroms,
consistent with a rough Guinier Rg of 3.61 nm versus the deposited 3.6 nm;
the raw file does not explicitly declare units. The assumption and selected
points are recorded alongside results in `outputs/SASDNV6/`. `--q-max` is in
inverse angstroms; `--q-units` controls conversion of the native input.
Only q and intensity enter the model; noise SD is inferred. There is no
synthetic-noise addition, averaging, or intensity clipping. The default 0–10 nm
pair-distance display is retained, while moment integrals use the full 40 nm
grid. The deposited Dmax is 17.5 nm, beyond that display window. Published Rg
is an analysis of the same experimental data, not independent ground truth.

## Guinier uncertainty from the saved GP

Run `MPLCONFIGDIR=IDPs/.mplconfig IDPs/.venv/bin/python IDPs/guinier_uq.py`
to create `outputs/SASDNV6/guinier_uq.png` and its JSON report, without new
inference. It projects log GP mean onto `a + b q²` using ordinary least
squares, with q in inverse nm. It propagates the full marginalized latent GP
covariance through log, the linear regression, and `Rg = sqrt(-3 b)` using
first-order Jacobians. No independent-point assumption or regression residual
variance is substituted for the GP covariance. The default fixed cutoff is
0.036 inverse Å; the third panel shows sensitivity to wider/narrower windows.
Intervals exclude Guinier approximation bias and cutoff-selection uncertainty.
This is a summary of the GP trained on the full fitted q range, not a new
independent low-q-only inference. The 3.6 nm line is the SASDNV6 reference.
