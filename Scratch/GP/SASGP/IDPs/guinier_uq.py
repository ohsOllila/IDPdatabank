"""Guinier regression of a saved GP posterior, with first-order covariance propagation.

No inference is rerun. OLS defines the straight-line projection; its uncertainty
comes from the full GP covariance, not regression residuals or independent errors.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def guinier(q_per_A, mean, covariance, q_max_per_A):
    use = np.asarray(q_per_A) <= q_max_per_A
    q = np.asarray(q_per_A)[use]*10  # inverse nm
    m = np.asarray(mean)[use]
    C = np.asarray(covariance)[np.ix_(use, use)]
    if len(q) < 3 or np.any(m <= 0) or not np.isfinite(m).all():
        raise ValueError('Require >=3 selected points and positive finite GP means')
    X = np.column_stack((np.ones(len(q)), q*q))
    A = np.linalg.pinv(X)
    log_mean = np.log(m)
    log_cov = C/np.outer(m, m)  # Jacobian of log: diag(1/m)
    beta = A @ log_mean
    beta_cov = A @ log_cov @ A.T
    if beta[1] >= 0:
        raise ValueError('Guinier slope must be negative')
    rg = np.sqrt(-3*beta[1])
    rg_sd = 3/(2*rg)*np.sqrt(max(beta_cov[1,1], 0))
    report = dict(q_max_requested_per_A=q_max_per_A, points=len(q),
                  q_range_per_A=[float(q.min()/10),float(q.max()/10)],
                  intercept=float(beta[0]), slope_nm2=float(beta[1]),
                  coefficient_covariance=beta_cov.tolist(), Rg_nm=float(rg),
                  Rg_sd_nm=float(rg_sd), interval_95_nm=[float(rg-1.96*rg_sd),float(rg+1.96*rg_sd)],
                  max_q_Rg=float(q.max()*rg),
                  max_fractional_GP_sd=float(np.max(np.sqrt(np.maximum(np.diag(C),0))/m)),
                  slope_nonnegative_z=float(-beta[1]/np.sqrt(max(beta_cov[1,1],np.finfo(float).tiny))))
    return report, q*q, log_mean, log_cov, X@beta, X@beta_cov@X.T


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'outputs/SASDNV6')
    parser.add_argument('--q-max',type=float,default=.036,help='Fixed Guinier upper cutoff in inverse Å')
    args=parser.parse_args()
    p=args.output_dir
    data=np.loadtxt(p/'saxs_gp_intensity.csv',delimiter=',',skiprows=1)
    saved=np.load(p/'saxs_gp_covariance.npz')
    np.testing.assert_allclose(data[:,0],saved['q_per_A'])
    q,m=data[:,0],data[:,2]
    report,x,logm,logcov,line,linecov=guinier(q,m,saved['covariance_q'],args.q_max)
    sensitivity=[guinier(q,m,saved['covariance_q'],cut)[0] for cut in sorted(set([.022,.025,.028,.036,args.q_max]))]
    result=dict(method='OLS projection of log GP mean; first-order propagation of full marginalized GP covariance',
                formulas=['C_log = diag(1/m) C_I diag(1/m)', 'A = (X.T X)^(-1) X.T; beta = A log(m)',
                          'C_beta = A C_log A.T', 'Rg = sqrt(-3 b); Var(Rg) = 9 Var(b)/(4 Rg^2)'],
                primary=report, cutoff_sensitivity=sensitivity,
                limitations='Fixed-window first-order approximation. Excludes Guinier truncation bias and cutoff-selection uncertainty. GP was trained on q <= 0.15 inverse Å; this is not an independent fit to only the low-q observations.',
                reference='https://journals.iucr.org/j/issues/2016/05/00/vg5047/')
    (p/'guinier_uq.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(17,4.8),constrained_layout=True)
    sd=np.sqrt(np.maximum(np.diag(logcov),0))
    axes[0].plot(x,logm,label='Log GP mean')
    axes[0].fill_between(x,logm-1.96*sd,logm+1.96*sd,alpha=.2,label='95% linearized GP interval')
    axes[0].plot(x,line,'--',color='C1',label='Guinier line (OLS)')
    axes[0].set(xlabel='q² (nm⁻²)',ylabel='ln I(q)',title=f'Guinier window: q ≤ {args.q_max:g} Å⁻¹')
    gx=np.linspace(report['Rg_nm']-4*report['Rg_sd_nm'],report['Rg_nm']+4*report['Rg_sd_nm'],401)
    density=np.exp(-.5*((gx-report['Rg_nm'])/report['Rg_sd_nm'])**2)/(report['Rg_sd_nm']*np.sqrt(2*np.pi))
    axes[1].plot(gx,density,label='Guinier: first-order UQ')
    axes[1].axvspan(*report['interval_95_nm'],alpha=.2)
    previous=json.loads((p/'saxs_gp_fit.json').read_text())['radius_of_gyration']
    axes[1].axvline(previous['median_nm'],color='C1',label='P(r) moment median')
    axes[1].axvspan(*previous['interval_95_nm'],color='C1',alpha=.12,label='P(r) moment 95% interval')
    axes[1].axvline(3.6,color='C3',ls='--',label='SASDNV6 deposited Guinier 3.6 nm')
    axes[1].set(xlabel='Rg (nm)',ylabel='Probability density (nm⁻¹)',title=f"Guinier Rg = {report['Rg_nm']:.3f} ± {report['Rg_sd_nm']:.3f} nm (SD)")
    axes[2].errorbar([z['q_max_requested_per_A'] for z in sensitivity],
                     [z['Rg_nm'] for z in sensitivity],yerr=[1.96*z['Rg_sd_nm'] for z in sensitivity],fmt='o-',capsize=4,label='95% first-order intervals')
    axes[2].axhline(3.6,color='C3',ls='--',label='Deposited Guinier')
    axes[2].set(xlabel='Upper q cutoff (Å⁻¹)',ylabel='Rg (nm)',title='Cutoff sensitivity (same GP)')
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.savefig(p/'guinier_uq.png',dpi=180)
    plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
