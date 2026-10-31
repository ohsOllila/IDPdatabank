# SAXS quality-metric toy problem: GP CRPS vs BIFT chi-squared

As part of deciding how the IDPdatabank should score SAXS trajectories, this
toy problem compares the current pipeline (BIFT-processed data + reduced
chi-squared) with a proposed alternative (sinc-transform GP regression of the
experiment, simulations scored by CRPS in units of the GP's predictive
uncertainty).

## Design

A synthetic experiment with known ground truth. The truth is an analytic
two-Gaussian coil P(r) matched to SASDQK7's fitted Rg, I(0), and noise level;
20 "simulations" carry controlled perturbations (size scaling, shoulder
weight, tail length/tilt, compact bumps, broadening, truncation), giving an
exact true ranking. Five error-bar/data scenarios test each method's weak
spot: calibrated bars (control), q-shaped miscalibrated bars (proportional to
I against constant-noise data), low-q-inflated bars, heteroscedastic low-q
noise, and a smooth correlated low-q systematic. Each scenario runs 8
independent noise realizations. Scoring goes through the **actual compiled
BIFT binary** (both its constant and non-constant error-correction modes)
using our notebook's chi-squared convention -- the full IFT machinery is in
the loop.

## Results

Median Spearman rho vs the true ranking over 8 realizations (top-5 overlap
with the true best-5 in parentheses):

| Scenario                | BIFT (const) | BIFT (non-const) | GP CRPS    |
|------------------------|--------------|------------------|------------|
| calibrated             | 0.96 (0.7)   | 0.95 (0.8)       | 0.96 (0.8) |
| q-shaped bars          | 0.64 (0.3)   | 0.55 (0.2)       | 0.93 (0.6) |
| low-q bars inflated    | 0.92 (0.8)   | 0.94 (0.8)       | 0.97 (0.8) |
| noisy low-q            | 0.82 (0.5)   | 0.77 (0.4)       | 0.84 (0.6) |
| systematic error       | 0.67 (0.5)   | 0.67 (0.4)       | 0.71 (0.5) |

## Takeaways

1. **A structural difference in requirements:** BIFT *requires* an error bar
   for every point -- its reader rejects sigma <= 0, and with proportional
   bars on a decaying curve it crashes outright unless sigma is floored. The
   GP operates without reported error bars entirely; it infers the noise
   level from the data scatter, so it is applicable to datasets with missing
   or placeholder uncertainties (several in the databank, including SASDQK7,
   whose sigma column is placeholder values).
2. When error bars are q-dependently wrong, chi-squared ranking degrades
   badly (0.64) while the GP holds (0.93). BIFT's own q-dependent error
   correction makes ranking *worse* (0.55) -- it calibrates sigma so BIFT's
   own fit looks statistically perfect, which is circular.
3. With honest bars, heteroscedastic noise, or correlated systematics, the
   two methods are statistically indistinguishable -- and both degrade under
   systematics (rho ~ 0.7), which bounds how much any pointwise scoring can
   be trusted.

## Caveats

The truth is from the GP's own model family (favorable to it); one dataset
shape; the GP's constant-noise assumption is a real assumption, with a
q-dependent noise model the natural upgrade.

Code, tests, and per-scenario figures are in `Scratch/GP/SASGP/IDPs/` on
branch `sax_gp_idps` (PR #37); methodology is in that directory's README.
