"""Linear sinc-transform GP with a smooth, zero-at-both-ends prior."""
import json
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def gyration_samples(r, weights, mean, covariance, samples=50000, seed=42):
    """Propagate the joint GP covariance to M0, M2 and Rg in nm.

    Guinier: I(q)/I(0) = 1 - q² Rg²/3 + O(q⁴).
    Expanding sinc gives Rg² = M2/(2 M0). Sample the two linear
    moments jointly, exactly equivalent to integrating full GP draws.
    """
    operator = np.vstack((weights, weights*(r/10)**2))
    moment_mean = operator @ mean
    moment_cov = operator @ covariance @ operator.T
    moment_cov = (moment_cov + moment_cov.T)/2
    eigenvalues, vectors = np.linalg.eigh(moment_cov)
    tolerance = 1e-10*max(np.max(np.abs(eigenvalues)), np.finfo(float).tiny)
    if eigenvalues.min() < -tolerance:
        raise ValueError('Moment covariance is not positive semidefinite')
    draws = moment_mean + np.random.default_rng(seed).standard_normal((samples, 2)) @ (
        vectors*np.sqrt(np.maximum(eigenvalues, 0))).T
    valid = (draws[:, 0] > 0) & (draws[:, 1] >= 0)
    rg = np.full(samples, np.nan)
    rg[valid] = np.sqrt(draws[valid, 1]/(2*draws[valid, 0]))
    return draws, rg, valid


def fit_saxs(q, intensity, output_dir, r_max=400., points=401,
             inference='nuts', prior_fn='uniform', samples=500, warmup=500, chains=2, seed=43, reference_rg_nm=None, guinier_q_max=None):
    """Fit intensities with an inferred constant observation-noise SD.

    P includes the intensity scale: I(q) = integral P(r) sinc(qr) dr.
    Zero mean, no reflection, no positivity or normalization constraint.
    Intervals marginalize kernel hyperparameters using membrane-style NUTS or Laplace.
    """
    q, y = np.asarray(q, float), np.asarray(intensity, float)
    if (q.ndim != 1 or y.shape != q.shape or len(q) < 2
            or not np.isfinite(q).all() or not np.isfinite(y).all()
            or np.any(q < 0) or np.any(np.diff(q) <= 0)
            or not np.isfinite(r_max) or r_max <= 0 or points < 51):
        raise ValueError('Require sorted nonnegative q, finite data, positive r_max and >=51 grid points')
    r = np.linspace(0, r_max, points)
    dr = r[1]
    weights = np.full(points, dr)
    weights[[0, -1]] *= .5
    H = np.sinc(np.outer(q, r) / np.pi) * weights
    scale = float(np.max(np.abs(y)))
    if scale <= 0:
        raise ValueError('Intensity scale must be nonzero')
    yn = y / scale
    # Used only for optimizer initialization; noise is inferred from the likelihood.
    sn = np.clip(np.median(np.abs(np.diff(yn)-np.median(np.diff(yn))))/.67449/np.sqrt(2), 1e-5, .5)
    distance2 = (r[:, None] - r[None, :]) ** 2
    # R <= r_max/4 makes the envelope at the integration boundary <= exp(-16).
    bounds = np.log([(1e-5/r_max, 100/r_max), (2*dr, r_max/2),
                     (2*dr, r_max/4), (4*dr, r_max/4), (1e-6, 1.)])

    def covariance(logtheta):
        amplitude, ell, r0, R = np.exp(logtheta[:4])
        w = amplitude * (-np.expm1(-(r/r0)**2)) * np.exp(-(r/R)**2)
        return w[:, None]*w[None, :]*np.exp(-distance2/(2*ell**2)), w

    def objective(theta):
        K, _ = covariance(theta)
        C = H @ K @ H.T + (np.exp(theta[4])**2 + 1e-12)*np.eye(len(q))
        factor = cho_factor(C, lower=True)
        return .5*yn @ cho_solve(factor, yn) + np.log(np.diag(factor[0])).sum() + .5*len(q)*np.log(2*np.pi)

    starts = [[10/r_max, r_max/20, r_max/25, r_max*f, sn] for f in (.10, .18, .24)]
    fits = [minimize(objective, np.clip(np.log(start), bounds[:, 0], bounds[:, 1]),
                     method='L-BFGS-B', bounds=bounds, options={'maxiter': 150}) for start in starts]
    valid = [fit for fit in fits if fit.success and np.isfinite(fit.fun)]
    if not valid:
        raise RuntimeError('GP optimization failed: ' + '; '.join(str(f.message) for f in fits))
    best = min(valid, key=lambda fit: fit.fun)
    output_dir.mkdir(parents=True, exist_ok=True)
    from bayesian_saxs import sample_hyperparameters, mixture_summary
    theta_draws, inference_info, physical_chain = sample_hyperparameters(
        r, H, yn, np.exp(bounds), np.exp(best.x), output_dir, scale,
        inference=inference, prior_fn=prior_fn, samples=samples,
        warmup=warmup, chains=chains, seed=seed)
    r_means, r_variances, q_means, q_variances = [], [], [], []
    moment_draws, rg_draws, valid_draws = [], [], []
    cov_sum = np.zeros((points, points))
    prior_variance = np.zeros(points)
    for index, theta in enumerate(theta_draws):
        K, w = covariance(np.log(theta))
        cross = K @ H.T
        C = H @ cross + (theta[4]**2+1e-12)*np.eye(len(q))
        factor = cho_factor(C, lower=True)
        mean = cross @ cho_solve(factor, yn)*scale
        posterior = (K-cross @ cho_solve(factor, cross.T))*scale**2
        posterior = (posterior+posterior.T)/2
        r_means.append(mean)
        r_variances.append(np.maximum(np.diag(posterior), 0))
        q_means.append(H @ mean)
        q_variances.append(np.maximum(np.einsum('ij,ij->i', H @ posterior, H), 0))
        cov_sum += posterior
        prior_variance += (w*scale)**2
        moments, rg, rg_valid = gyration_samples(r, weights, mean, posterior, samples=50, seed=seed+index+1)
        moment_draws.append(moments)
        rg_draws.append(rg)
        valid_draws.append(rg_valid)
    mean, sd, r_interval, r_within, r_between = mixture_summary(r_means, r_variances)
    predicted, intensity_sd, q_interval, q_within, q_between = mixture_summary(q_means, q_variances)
    r_means = np.asarray(r_means)
    total_covariance = cov_sum/len(theta_draws) + (r_means-mean).T @ (r_means-mean)/len(theta_draws)
    np.savez_compressed(output_dir/'saxs_gp_covariance.npz', r_A=r, q_per_A=q,
                        covariance_r=total_covariance, covariance_q=H @ total_covariance @ H.T,
                        within_variance_r=r_within, between_variance_r=r_between,
                        within_variance_q=q_within, between_variance_q=q_between)
    np.savetxt(output_dir/'saxs_gp_real_space.csv',
               np.column_stack((r, mean, sd, np.sqrt(prior_variance/len(theta_draws)), r_interval.T)),
               delimiter=',', header='r_A,P_mean,P_sd,hyperposterior_envelope_rms,P_lower_95,P_upper_95', comments='')
    np.savetxt(output_dir/'saxs_gp_intensity.csv', np.column_stack((q, y, predicted, intensity_sd,
               np.sqrt(intensity_sd**2+np.mean((theta_draws[:,4]*scale)**2)), q_interval.T)), delimiter=',',
               header='q_per_A,noisy_I,I_mean,I_latent_sd,I_predictive_sd,I_lower_95,I_upper_95', comments='')
    amplitude, ell, r0, R, noise_sigma = np.median(theta_draws, axis=0)
    moments, rg, rg_valid = np.concatenate(moment_draws), np.concatenate(rg_draws), np.concatenate(valid_draws)
    rg_quantiles = np.quantile(rg[rg_valid], [.025, .5, .975]) if rg_valid.any() else None
    np.savetxt(output_dir/'saxs_gp_rg_samples.csv',
               np.column_stack((moments, rg, rg_valid.astype(int), np.repeat(np.arange(len(theta_draws)), 50))),
               delimiter=',', header='I0_au,M2_au_nm2,Rg_nm,valid_moments,hyperparameter_draw', comments='')
    report = dict(amplitude=float(amplitude*scale), ell_A=float(ell), r0_A=float(r0), R_A=float(R),
                  r_max_A=float(r_max), grid_points=points, noise_sigma=float(noise_sigma*scale),
                  noise_model="Inferred constant Gaussian observation SD",
                  noise_sigma_interval_95=(np.quantile(theta_draws[:,4], [.025,.975])*scale).tolist(),
                  negative_log_marginal_likelihood_scaled=float(best.fun), optimizer_message=str(best.message),
                  successful_starts=len(valid),
                  mle_near_bound_parameters=[name for name, x, (lo, hi) in zip(
                      ['amplitude', 'ell', 'r0', 'R', 'noise_sigma'], best.x, bounds) if min(x-lo, hi-x) < .01],
                  inference=inference_info, parameter_summary='Posterior medians',
                  intervals='Pointwise equal-tailed hyperparameter-mixture intervals; no positivity constraint',
                  mean='Zero', transform='I(q) = integral P(r) sinc(qr) dr; sinc(x)=sin(x)/x')
    report['radius_of_gyration'] = dict(
        formula='Rg^2 = integral r^2 P(r) dr / (2 integral P(r) dr)',
        median_nm=float(rg_quantiles[1]) if rg_quantiles is not None else None,
        interval_95_nm=rg_quantiles[[0, 2]].tolist() if rg_quantiles is not None else None,
        draws=len(rg), hyperparameter_draws=len(theta_draws), conditional_draws_per_hyperparameter=50, seed=seed, invalid_moment_fraction=float(1-rg_valid.mean()),
        uncertainty='Hyperparameter-marginalized joint GP moments, conditional on valid moments; no pointwise positivity constraint')
    (output_dir/'saxs_gp_fit.json').write_text(json.dumps(report, indent=2)+'\n')
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    axes[0].plot(q, y, '.', alpha=.5, label='Observed intensity')
    axes[0].plot(q, predicted, label='FTGP mean')
    axes[0].fill_between(q, q_interval[0], q_interval[1],
                         alpha=.25, label='95% latent interval')
    axes[0].set(xlabel='q (Å⁻¹)', ylabel='Intensity (a.u.)', title='SAXS fit')
    # Convert the density as well as distance to preserve its integral.
    axes[1].plot(r/10, mean*10, label='FTGP mean')
    axes[1].fill_between(r/10, r_interval[0]*10, r_interval[1]*10,
                         alpha=.25, label='95% interval')
    axes[1].set(xlabel='Pair separation r (nm)', ylabel='P(r) (a.u. / nm)',
                xlim=(0, 25), title='Pair-distance distribution')
    if rg_quantiles is not None:
        axes[2].hist(rg[rg_valid], bins=70, density=True, alpha=.65, label='P(r) moments: posterior draws')
        axes[2].axvspan(rg_quantiles[0], rg_quantiles[2], alpha=.15, color='C1', label='P(r) moments: 95% interval')
        axes[2].axvline(rg_quantiles[1], color='C1', label=f'P(r) median {rg_quantiles[1]:.3f} nm')
    else:
        axes[2].text(.5, .5, 'No valid Rg draws', ha='center', transform=axes[2].transAxes)
    axes[2].set(xlabel='Radius of gyration (nm)', ylabel='Probability density (nm⁻¹)',
                title=f'Rg uncertainty ({inference.upper()})')
    if guinier_q_max is not None:
        from guinier_uq import guinier
        guinier_result, *_ = guinier(q, predicted, H @ total_covariance @ H.T, guinier_q_max)
        center, width = guinier_result['Rg_nm'], guinier_result['Rg_sd_nm']
        gx = np.linspace(center-4*width, center+4*width, 401)
        density = np.exp(-.5*((gx-center)/width)**2)/(width*np.sqrt(2*np.pi))
        axes[2].plot(gx, density, color='C2', linewidth=2,
                     label=f'Guinier 1st-order UQ: {center:.3f} nm')
        axes[2].axvspan(*guinier_result['interval_95_nm'], color='C2', alpha=.12,
                        label=f'Guinier 95% (q ≤ {guinier_q_max:g} Å⁻¹)')
        axes[2].set_title('Rg: moment propagation vs Guinier')
    if reference_rg_nm is not None:
        axes[2].axvline(reference_rg_nm, color='C3', linestyle='--', linewidth=1.8,
                        label=f'Guinier reference {reference_rg_nm:g} nm')
    if not rg_valid.all():
        axes[2].text(.02, .95, f'Invalid moments: {1-rg_valid.mean():.1%}',
                     va='top', transform=axes[2].transAxes)
    if inference == 'nuts' and not inference_info['diagnostic_checks_passed']:
        fig.suptitle('Provisional: NUTS diagnostic checks have not passed')
    for ax in axes:
        ax.axhline(0, color='0.5', lw=.7)
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=.2)
    fig.savefig(output_dir/'saxs_gp_fit.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(5, 2, figsize=(11, 12), constrained_layout=True)
    for j, label in enumerate(['Amplitude (a.u. / Å)', 'ell (Å)', 'r0 (Å)', 'R (Å)', 'Noise SD (a.u.)']):
        for c, chain_values in enumerate(physical_chain):
            axes[j, 0].plot(chain_values[:, j], alpha=.7, lw=.6, label=f'Chain {c+1}')
        axes[j, 0].set(ylabel=label, xlabel='Posterior draw')
        axes[j, 1].hist(physical_chain[:, :, j].ravel(), bins=40, density=True)
        axes[j, 1].set(xlabel=label, ylabel='Density')
    axes[0, 0].legend()
    fig.suptitle(f'{inference.upper()} hyperparameter posterior')
    fig.savefig(output_dir/'saxs_gp_hyperparameters.png', dpi=150)
    plt.close(fig)
    return report
