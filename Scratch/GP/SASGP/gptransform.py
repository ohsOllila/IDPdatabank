import torch as torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
import scipy
import time
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from dataclasses import dataclass


# Computes L_2 norm of the two lists of vectors x1 and x2
def custom_cdist(x1, x2):
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x1_pad = torch.ones_like(x1_norm)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    x2_pad = torch.ones_like(x2_norm)
    x1_ = torch.cat([-2. * x1, x1_norm, x1_pad], dim=-1)
    x2_ = torch.cat([x2, x2_pad, x2_norm], dim=-1)
    res = x1_.matmul(x2_.transpose(-2, -1))
    res.clamp_min_(0)
    return res


# Non-stationary amplitude kernel with constant lengthscale and input-dependent width
def f_kernel(Xdd, ell, σ1, σ2):
    prefactor = torch.outer(σ1, σ2)
    exponential = torch.exp(torch.clamp(-(Xdd) / (2 * ell ** 2), min=(-100), max=0))
    K = prefactor * exponential
    return K


# Patrik membrane FT functions
# Cosine "forward transform": ed(r) -> ff(q)
def ed2ff(r, ed, q):
    """
    Cosine transform from r-space to q-space.
    Returns: ff (Nq,)
    """
    r1 = r.reshape(-1)
    f1 = ed.reshape(-1)
    q1 = q.reshape(-1)

    dr = r1[1] - r1[0]
    K = torch.cos(torch.outer(q1, r1))          # (Nq, Nr)
    ff = torch.trapz(K * f1[None, :], dx=dr, dim=1)  # (Nq,)
    return ff


def ff2ed(q, ff, r):
    """
    Inverse cosine transform from q-space to r-space.
    Returns: ed (Nr,)
    """
    q1 = q.reshape(-1)
    F1 = ff.reshape(-1)
    r1 = r.reshape(-1)

    dq = q1[1] - q1[0]
    K = torch.cos(torch.outer(r1, q1))          # (Nr, Nq)
    ed = torch.trapz(K * F1[None, :], dx=dq, dim=1) / np.pi  # (Nr,)
    return ed


# ----------------------------
# Cosine transform (batched)
# ----------------------------
def ed2ff_batch(r, eds, q):
    """
    Batch cosine transform from r-space to q-space.

    Inputs
    ------
    r    : (Nr,) or (Nr,1)
    eds  : (B, Nr) or (B, Nr, 1)   OR   iterable of (Nr,) tensors
    q    : (Nq,) or (Nq,1)

    Returns
    -------
    ffs : (B, Nq)
    """
    r1 = r.reshape(-1)
    q1 = q.reshape(-1)
    dr = r1[1] - r1[0]

    # Make eds a (B, Nr) tensor
    if isinstance(eds, (list, tuple)):
        F = torch.stack([ri.reshape(-1) for ri in eds], dim=0)
    else:
        F = eds
        if F.ndim == 3 and F.shape[-1] == 1:
            F = F[..., 0]
        F = F.reshape(F.shape[0], -1)  # (B, Nr)

    # Kernel: (Nq, Nr)
    K = torch.cos(torch.outer(q1, r1))

    # Broadcast multiply: (B, Nq, Nr) then integrate over r (dim=2)
    ffs = torch.trapz(K[None, :, :] * F[:, None, :], dx=dr, dim=2)  # (B, Nq)
    return ffs


def ff2ed_batch(q, ffs, r):
    """
    Batch inverse cosine transform from q-space to r-space.

    Inputs
    ------
    q    : (Nq,) or (Nq,1)
    ffs  : (B, Nq) or (B, Nq, 1)   OR   iterable of (Nq,) tensors
    r    : (Nr,) or (Nr,1)

    Returns
    -------
    eds : (B, Nr)
    """
    q1 = q.reshape(-1)
    r1 = r.reshape(-1)
    dq = q1[1] - q1[0]

    # Make ffs a (B, Nq) tensor
    if isinstance(ffs, (list, tuple)):
        S = torch.stack([si.reshape(-1) for si in ffs], dim=0)
    else:
        S = ffs
        if S.ndim == 3 and S.shape[-1] == 1:
            S = S[..., 0]
        S = S.reshape(S.shape[0], -1)  # (B, Nq)

    # Kernel: (Nr, Nq)
    K = torch.cos(torch.outer(r1, q1))

    # Broadcast multiply: (B, Nr, Nq) then integrate over q (dim=2)
    eds = torch.trapz(K[None, :, :] * S[:, None, :], dx=dq, dim=2) / np.pi  # (B, Nr)
    return eds


@dataclass
class FitMetrics:
    rmse: float
    chi2: float
    coverage: float
    nlpd: float


def compute_metrics(mu, cov, y_true, coverage_sigma=2.0):
    """
    Evaluate GP posterior against ground truth.

    mu            : (N,1) or (N,)  posterior mean
    cov           : (N,N)          posterior covariance
    y_true        : (N,1) or (N,)  ground truth
    coverage_sigma: half-width of CI in std units (default 2 → ~95%)
    """
    mu_flat = mu.reshape(-1).detach()
    y_flat = y_true.reshape(-1).detach()
    std = torch.diag(cov).detach().clamp(min=0).sqrt()

    residuals = mu_flat - y_flat
    rmse = residuals.pow(2).mean().sqrt().item()
    chi2 = (residuals / std.clamp(min=1e-10)).pow(2).mean().item()

    in_band = (y_flat >= mu_flat - coverage_sigma * std) & (y_flat <= mu_flat + coverage_sigma * std)
    coverage = in_band.float().mean().item()

    var = std.pow(2).clamp(min=1e-10)
    nlpd = 0.5 * (torch.log(2 * torch.tensor(np.pi) * var) + residuals.pow(2) / var).mean().item()

    return FitMetrics(rmse=rmse, chi2=chi2, coverage=coverage, nlpd=nlpd)


def _is_pd(A):
    try:
        torch.linalg.cholesky(A)
        return True
    except torch.linalg.LinAlgError:
        return False


class nearestPDClass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        # Symmetrize A to ensure it's symmetric
        A_sym = (A + A.t()) / 2

        # Check for PD to see if thats all it took
        if _is_pd(A_sym):
            return A_sym

        # Perform eigenvalue decomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(A_sym)

        # Reconstruct the matrix
        A_pos_def = (eigenvectors @ torch.diag(eigenvalues)) @ eigenvectors.t()
        condition = _is_pd(A_pos_def)
        mineigval = torch.min(eigenvalues)
        i = 0
        while not condition:
            A_pos_def = eigenvectors @ torch.diag(eigenvalues + 1 / (2 * 10 ** (15 - i)) - mineigval) @ eigenvectors.t()
            i += 1
            condition = _is_pd(A_pos_def)
        return A_pos_def

    @staticmethod
    def backward(ctx, grad_output):
        # Output is always symmetric; symmetrize gradient accordingly
        return (grad_output + grad_output.T) / 2


class data(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        if len(self.X) != len(self.Y):
            raise Exception("len(X) != len(Y)")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        _x = self.X[index].unsqueeze(dim=0)
        _y = self.Y[index].unsqueeze(dim=0)
        return _x, _y


def train_loop(dataloader, model, optimizer, totalEpochs, r_grid, q_train, sq_train, q_infer, r_infer, ylo_q, yhi_q, plot=True):
    model.train()
    optimizer.zero_grad()
    losses = []
    tic = time.time()
    last_report_loss = None
    consecutive_below = 0
    for epoch in range(totalEpochs):
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            loss = model.NEG_LMLH_Trapz(r_grid, X[0], y[0])

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.detach().item())

        if epoch % 25 == 0:
            toc = time.time()
            average_loss = np.mean(np.array(losses[-int(len(dataloader) * 10):]))
            print(f"Average loss: {average_loss:>7f}  [{epoch:>5d}/{totalEpochs:>5d}]")
            if last_report_loss is not None and abs(last_report_loss - average_loss) < 0.01:
                consecutive_below += 1
                if consecutive_below >= 3:
                    print(f"Early stopping: loss change < 0.01 for 3 consecutive reports")
                    break
            else:
                consecutive_below = 0
            last_report_loss = average_loss
            model.print_params()
            print(f"Minutes Taken Since Last Report: {(toc - tic) / 60:>4f} ")
            print()

            if plot:
                with torch.no_grad():
                    plt.title("F(q) Prediction")
                    μ_q, Σ_q = model.predict_sq_trapz(r_grid, q_infer, q_train, sq_train)
                    plt.scatter(q_train, sq_train, alpha=0.2, label="Presented Data")
                    plt.plot(q_infer.detach().numpy(), μ_q.detach().numpy(), label="Mean Prediction")
                    plt.fill_between(q_infer.T[0].detach().numpy(), μ_q.T[0].detach().numpy() + torch.diag(Σ_q).detach().numpy() ** 0.5, μ_q.T[0].detach().numpy() - torch.diag(Σ_q).detach().numpy() ** 0.5, alpha=0.5, label="1 +- std")
                    plt.ylim(ylo_q, yhi_q)
                    plt.xlim(q_infer[0], q_infer[-1])
                    plt.legend()
                    plt.show()

                plt.figure(figsize=(6, 4))
                plt.plot(losses, label='Loss')
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title("Loss Curve")
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.show()

            tic = time.time()

    return losses


def real_space_mean(r, theta, mean_fn, start):
    """Evaluate the shared real-space mean; cosine B is a length scale."""
    r = r.reshape(-1)
    if mean_fn == "cosine":
        A, B = theta[start:start + 2]
        return -A * torch.cos(np.pi * r / B) * (torch.abs(r / B) <= 1.5)
    if mean_fn == "tophat":
        A, B, C, D = theta[start:start + 4]
        distance = torch.abs(r)
        return (-A * (distance <= B) + C * ((distance >= B) & (distance <= B + D)))
    return torch.zeros_like(r)


class GP(nn.Module):
    # Constructs GP object and establishes initial parameters of model
    def __init__(self, init_params, init_param_bounds, num_bonds, rho_init, SAMPLE_COUNT_init,
                 noise_model='quadratic', prior_fn='uniform', mean_fn='zero'):
        """
        noise_model : 'constant'  ->  params: [ell, c, k, a, sigma_n, q_base]           (6)
                      'quadratic' ->  params: [ell, c, k, a, sigma_n_base, sigma_n_slope, q_base] (7)
        prior_fn    : 'uniform'   ->  uniform over bounded space (log Jacobian of logit)
                      'gaussian'  ->  N(0, 3^2) on theta_raw
        mean_fn     : 'zero'      ->  zero mean
                      'cosine'    ->  appends [A, B] to theta; mean_r = -A·cos(π·r/B) * (|r/B|≤1.5)
                      'tophat'    ->  appends [A, B, C, D] to theta; center -A, sides +C
        init_params / init_param_bounds must include mean params at the end when mean_fn != 'zero'.
        Use build_bounds(..., mean_fn=) in run.py to construct bounds with the right size.
        """
        super().__init__()

        assert noise_model in ('constant', 'quadratic'), \
            f"noise_model must be 'constant' or 'quadratic', got '{noise_model}'"
        assert prior_fn in ('uniform', 'gaussian'), \
            f"prior_fn must be 'uniform' or 'gaussian', got '{prior_fn}'"
        assert mean_fn in ('zero', 'cosine', 'tophat'), \
            f"mean_fn must be 'zero', 'cosine', or 'tophat', got '{mean_fn}'"
        self.noise_model = noise_model
        self.prior_fn = prior_fn
        self.mean_fn = mean_fn
        self.construct_params(init_params, init_param_bounds)
        self.num_bonds = num_bonds
        self.nearestPD = nearestPDClass.apply
        self.rho = rho_init
        self.SAMPLE_COUNT = SAMPLE_COUNT_init

    def construct_params(self, init_params, init_params_bounds):
        # Ensure boundary conditions are met.
        for p in range(len(init_params)):
            if init_params[p] < init_params_bounds[p, 0] or init_params[p] > init_params_bounds[p, 1]:
                print("Parameter # " + str(p) + ", " + str(init_params[p]) + ", is not in the range [" + str(init_params_bounds[p, 0]) + ", " + str(init_params_bounds[p, 1]) + "]")
                raise Exception("Initial parameter outside of chosen boundary.")

        # Create tensor representation of the parameters
        self.theta_raw = nn.Parameter(
            scipy.special.logit((init_params - init_params_bounds[:, 0]) / (init_params_bounds[:, 1] - init_params_bounds[:, 0])), requires_grad=True
        )

        self.theta_bounds = init_params_bounds
        self.theta = (self.theta_bounds[:, 1] - self.theta_bounds[:, 0]) * torch.sigmoid(self.theta_raw) + self.theta_bounds[:, 0]

    def compute_params_from_raw(self):
        self.theta = (self.theta_bounds[:, 1] - self.theta_bounds[:, 0]) * torch.sigmoid(self.theta_raw) + self.theta_bounds[:, 0]
        return self.theta

    def log_uniform_prior(self):
        sig = torch.sigmoid(self.theta_raw)
        return torch.sum(torch.log(sig) + torch.log1p(-sig))

    def log_gaussian_prior(self, sigma=3.0):
        return -0.5 * torch.sum(self.theta_raw ** 2) / sigma ** 2

    def log_prior(self):
        if self.prior_fn == 'gaussian':
            return self.log_gaussian_prior()
        return self.log_uniform_prior()

    @property
    def _q_base_idx(self):
        """Index of q_base in theta: 5 for constant noise, 6 for quadratic."""
        return 5 if self.noise_model == 'constant' else 6

    @property
    def _mean_param_start_idx(self):
        return self._q_base_idx + 1

    @property
    def _n_mean_params(self):
        return {'zero': 0, 'cosine': 2, 'tophat': 4}[self.mean_fn]

    def print_params(self):
        params = self.compute_params_from_raw()
        print(f"l:             {params[0].item():>7f}")
        print(f"c:             {params[1].item():>7f}")
        print(f"k:             {params[2].item():>7f}")
        print(f"a:             {params[3].item():>7f}")
        if self.noise_model == 'quadratic':
            print(f"sigma_n_base:  {params[4].item():>7f}")
            print(f"sigma_n_slope: {params[5].item():>7f}")
        else:
            print(f"sigma_n:       {params[4].item():>7f}")
        print(f"q_base:        {params[self._q_base_idx].item():>7f}")
        i = self._mean_param_start_idx
        if self.mean_fn == 'cosine':
            print(f"mean_A:        {params[i].item():>7f}")
            print(f"mean_B:        {params[i+1].item():>7f}")
        elif self.mean_fn == 'tophat':
            print(f"mean_A:        {params[i].item():>7f}")
            print(f"mean_B:        {params[i+1].item():>7f}")
            print(f"mean_C:        {params[i+2].item():>7f}")
            print(f"mean_D:        {params[i+3].item():>7f}")

    # updated sigmoidal tophat kernel definition
    def K(self, r1, r2, ell, c, k, a):
        Xdd = custom_cdist(r1, r2)
        Xdd_Reversed = custom_cdist(-r1, r2)
        σ1 = self.width_fxn(r1.T[0], c, k, a)
        σ1_Reversed = self.width_fxn(-r1.T[0], c, k, a)
        σ2 = self.width_fxn(r2.T[0], c, k, a)
        return f_kernel(Xdd, ell, σ1, σ2) + f_kernel(Xdd_Reversed, ell, σ1_Reversed, σ2)

    # new width function worked up for Patrik membrane problem
    def width_fxn(self, x, c, k, a):
        # x: (N,) tensor
        z1 = torch.clamp(-k * (x - a), min=-100.0, max=25.0)
        z2 = torch.clamp(-k * (x + a), min=-100.0, max=25.0)
        return -c / (1.0 + torch.exp(z1)) + c / (1.0 + torch.exp(z2))

    def K_rr(self, r1, r2, adjust=True):
        Kdd = self.K(r1, r2, self.theta[0], self.theta[1], self.theta[2], self.theta[3])
        if adjust:
            return self.nearestPD(Kdd)
        return Kdd

    def K_rq(self, r1, r2, q2, adjust=True):
        Kdd = self.K(r1, r2, self.theta[0], self.theta[1], self.theta[2], self.theta[3])
        Krq = ed2ff_batch(r2, Kdd, q2)
        if adjust:
            return self.nearestPD(Krq)
        return Krq

    def K_qq(self, r1, r2, q1, q2, adjust=True):
        Kdd = self.K_rr(r1, r2, adjust=True)
        Kqr = ed2ff_batch(r2, Kdd, q2)
        Kqq = ed2ff_batch(r1, Kqr.T, q1).T
        if adjust:
            return self.nearestPD(Kqq)
        return Kqq

    def mean_r(self, r):
        return real_space_mean(r, self.theta, self.mean_fn,
                               self._mean_param_start_idx).reshape(r.shape)

    def mean_q(self, r, q):
        mean_r = self.mean_r(r)
        return ed2ff(r.T[0], mean_r.T[0], q.T[0]).unsqueeze(dim=1)

    def noise_q(self, q):
        q_flat = q.reshape(-1)
        base_noise = self.theta[4] ** 2 * torch.ones_like(q_flat)
        if self.noise_model == 'quadratic':
            base_noise = base_noise + self.theta[5] ** 2 * (q_flat ** 2)
        return base_noise

    def NEG_LMLH_Trapz(self, r_grid, q_train, sq_train):
        self.compute_params_from_raw()

        q_base = self.theta[self._q_base_idx]
        sq_train_shifted = sq_train - q_base

        mu_q = self.mean_q(r_grid, q_train)
        Kdd = self.K_qq(r_grid, r_grid, q_train, q_train)

        noise_diagonal = torch.diag(self.noise_q(q_train))
        Kdd += noise_diagonal

        L = torch.linalg.cholesky(Kdd)
        logdet = torch.linalg.slogdet(Kdd)[1]
        LMLH = 0.5 * (sq_train_shifted - mu_q).reshape(1, -1) @ (torch.cholesky_solve(sq_train_shifted - mu_q, L)) + 0.5 * logdet + (len(q_train) / 2) * np.log(2 * np.pi)

        return LMLH - self.log_prior()

    def predict_sq_trapz(self, r_grid, q_infer, q_train, sq_train, adjust=True):
        self.compute_params_from_raw()

        q_base = self.theta[self._q_base_idx]
        sq_train_shifted = sq_train - q_base

        Kii = self.K_qq(r_grid, r_grid, q_infer, q_infer, adjust=adjust)
        Kdd = self.K_qq(r_grid, r_grid, q_train, q_train, adjust=adjust)
        Kid = self.K_qq(r_grid, r_grid, q_infer, q_train, adjust=False)
        Kdi = Kid.T

        mu_q_dd = self.mean_q(r_grid, q_train)
        mu_q_ii = self.mean_q(r_grid, q_infer)

        noise_diagonal = torch.diag(self.noise_q(q_train))
        Kdd += noise_diagonal

        L = torch.linalg.cholesky(Kdd)

        # Predict on shifted data then add baseline back
        post_mean_shifted = mu_q_ii + Kid @ torch.cholesky_solve((sq_train_shifted - mu_q_dd).reshape(-1, 1), L)
        post_mean = post_mean_shifted + q_base

        post_cov = Kii - Kdi.T @ torch.cholesky_solve(Kdi, L)
        if adjust:
            return post_mean, self.nearestPD((post_cov + post_cov.T) / 2)
        return post_mean, post_cov

    def predict_ed_trapz(self, r_grid, r_infer, q_train, sq_train, adjust=True):
        self.compute_params_from_raw()

        # Intentionally use unshifted sq_train here: the constant q_base offset
        # in F(q) contributes a physical DC component to ρ(r) at r=0.
        # Subtracting it before the transform would remove that peak.
        # K_qq is built from the shifted GP (via predict_sq_trapz convention),
        # so the noise-weighted inversion is still consistent with training.
        Kii = self.K_rr(r_infer, r_infer, adjust=adjust)
        Kdd = self.K_qq(r_grid, r_grid, q_train, q_train, adjust=adjust)
        Kid = self.K_rq(r_infer, r_grid, q_train, adjust=False)
        Kdi = Kid.T

        mu_q_dd = self.mean_q(r_grid, q_train)
        mu_r_ii = self.mean_r(r_infer)

        noise_diagonal = torch.diag(self.noise_q(q_train))
        Kdd += noise_diagonal

        L = torch.linalg.cholesky(Kdd)

        post_mean = mu_r_ii + Kid @ torch.cholesky_solve((sq_train - mu_q_dd).reshape(-1, 1), L)
        post_cov = Kii - Kdi.T @ torch.cholesky_solve(Kdi, L)

        if adjust:
            return post_mean, self.nearestPD((post_cov + post_cov.T) / 2)
        return post_mean, post_cov

    # This is a method I was trying before I figured out about the float64 things and was using a log GP
    # on the g(r) so it was strictly positive. It worked but was super slow and gave almost the same predictions
    # as a standard GP on g(r). The speed made it very unscaleable and very hard to tune the hyper parameters due to
    # big gradients.
    def predict_ed_monte_carlo(self, r_grid, r_infer, q_infer, q_train, sq_train, optimal_shrinkage, return_untouched_data=False):
        self.compute_params_from_raw()

        q = torch.zeros((len(q_infer) + len(q_train)), 1)
        q[:len(q_infer)] = q_infer
        q[len(q_infer):] = q_train

        mu_r = self.mean_r(r_grid)
        Kdd_r = self.K_rr(r_grid, r_grid)

        MVN_r = MultivariateNormal(mu_r.T[0], Kdd_r)
        rsamples = MVN_r.rsample((self.SAMPLE_COUNT,)).double()  # Must be rsample so we can backprop

        qsamples = ed2ff_batch(r_grid, rsamples - 1, q)

        S_beta_alpha_q = None
        S_beta_alpha_r = None

        if optimal_shrinkage:
            cov = torch.cov(qsamples.T)

            mean = []
            var = []
            std = []
            skews = []
            kurtoses = []

            for index in range(len(q)):
                array = qsamples.T[index]
                mean.append(torch.mean(array))
                diffs = array - torch.mean(array)
                var.append(torch.mean(torch.pow(diffs, 2.0)))
                std_val = torch.pow(torch.mean(torch.pow(diffs, 2.0)), 0.5)
                std.append(std_val)
                zscores = diffs / std_val
                skews.append(torch.mean(torch.pow(zscores, 3.0)))
                kurtoses.append(torch.mean(torch.pow(zscores, 4.0)) - 3.0)

            elipitical_kurtoses = torch.mean(torch.tensor(kurtoses)) / 3
            print("Eliptical Kurtoses:", elipitical_kurtoses)

            mean_ev = (torch.trace(cov @ cov) / self.SAMPLE_COUNT) - (1 + elipitical_kurtoses) * (self.SAMPLE_COUNT / len(cov)) * (torch.trace(cov) / self.SAMPLE_COUNT) ** 2
            print("Mean of Eigenvalues:", mean_ev)

            sphericity = mean_ev / ((torch.trace(cov) / self.SAMPLE_COUNT) ** 2)
            print("Sphericity:", sphericity)
            print()

            β_0 = (sphericity - 1) / ((sphericity - 1) + elipitical_kurtoses * (2 * sphericity + self.SAMPLE_COUNT) / len(cov) + (sphericity + self.SAMPLE_COUNT) / (len(cov) - 1))
            α_0 = (1 - β_0) * torch.trace(cov) / self.SAMPLE_COUNT
            print("α_0:", α_0)
            print("β_0:", β_0)

            S_beta_alpha = β_0 * cov + α_0 * torch.eye(len(cov))
            print("isPD(S_beta_alpha):", _is_pd(S_beta_alpha))
            print("Frob Norm of Old Estimate with New:", torch.norm(S_beta_alpha - cov))
            print()

            cov = self.nearestPD(S_beta_alpha)
            S_beta_alpha_q = S_beta_alpha

        else:
            cov = self.nearestPD(torch.cov(qsamples.T))

        Kii = cov[:len(q_infer), :len(q_infer)]
        Kdd = cov[len(q_infer):, len(q_infer):]
        Kid = cov[:len(q_infer), len(q_infer):]
        Kdi = cov[len(q_infer):, :len(q_infer)]

        mu_q_dd = torch.mean(qsamples[:, len(q_infer):], dim=0).reshape(len(q_train), 1)
        mu_q_ii = torch.mean(qsamples[:, :len(q_infer)], dim=0).reshape(len(q_infer), 1)

        Kdd = Kdd + torch.eye(len(q_train)) * (self.theta[5] ** 2)

        L = torch.linalg.cholesky(Kdd)

        post_mean = mu_q_ii + Kid @ torch.cholesky_solve((sq_train - mu_q_dd).reshape(-1, 1), L)
        post_cov = Kii - Kdi.T @ torch.cholesky_solve(Kdi, L)

        MVN_post_q = MultivariateNormal(post_mean.T[0], self.nearestPD((post_cov + post_cov.T) / 2))
        qsamples_post = MVN_post_q.rsample((self.SAMPLE_COUNT,)).double()

        eds_post = ff2ed_batch(q_infer, qsamples_post, r_infer)

        if optimal_shrinkage:
            print("Computing real space post using optimal shrinkage.")

            cov = torch.cov(eds_post.T)

            mean = []
            var = []
            std = []
            skews = []
            kurtoses = []

            for index in range(len(r_infer)):
                array = eds_post.T[index]
                mean.append(torch.mean(array))
                diffs = array - torch.mean(array)
                var.append(torch.mean(torch.pow(diffs, 2.0)))
                std_val = torch.pow(torch.mean(torch.pow(diffs, 2.0)), 0.5)
                std.append(std_val)
                zscores = diffs / std_val
                skews.append(torch.mean(torch.pow(zscores, 3.0)))
                kurtoses.append(torch.mean(torch.pow(zscores, 4.0)) - 3.0)

            elipitical_kurtoses = torch.mean(torch.tensor(kurtoses)) / 3
            print("Eliptical Kurtoses:", elipitical_kurtoses)

            mean_ev = (torch.trace(cov @ cov) / self.SAMPLE_COUNT) - (1 + elipitical_kurtoses) * (self.SAMPLE_COUNT / len(cov)) * (torch.trace(cov) / self.SAMPLE_COUNT) ** 2
            print("Mean of Eigenvalues:", mean_ev)

            sphericity = mean_ev / ((torch.trace(cov) / self.SAMPLE_COUNT) ** 2)
            print("Sphericity:", sphericity)
            print()

            β_0 = (sphericity - 1) / ((sphericity - 1) + elipitical_kurtoses * (2 * sphericity + self.SAMPLE_COUNT) / len(cov) + (sphericity + self.SAMPLE_COUNT) / (len(cov) - 1))
            α_0 = (1 - β_0) * torch.trace(cov) / self.SAMPLE_COUNT
            print("α_0:", α_0)
            print("β_0:", β_0)

            S_beta_alpha = β_0 * cov + α_0 * torch.eye(len(cov))
            print("isPD(S_beta_alpha):", _is_pd(S_beta_alpha))
            print("Frob Norm of Old Estimate with New:", torch.norm(S_beta_alpha - cov))
            print()

            cov = self.nearestPD(S_beta_alpha)
            S_beta_alpha_r = S_beta_alpha

        else:
            cov = self.nearestPD(torch.cov(eds_post.T))

        mean = torch.mean(eds_post, dim=0).reshape(len(r_infer), 1)

        if return_untouched_data:
            return mean, self.nearestPD((cov + cov.T) / 2), S_beta_alpha_q, S_beta_alpha_r

        return mean, self.nearestPD((cov + cov.T) / 2)


class _NUTSModel:
    """Picklable Pyro model for NUTS multi-chain sampling."""

    def __init__(self, P, device_str, bounds, q1, r_flat, sq_t,
                 C, Xdd, Xdd_rev, eye_q,
                 noise_model, q_base_idx, mean_fn, mean_param_start, prior_fn="gaussian"):
        self.P = P
        self.device_str = device_str
        self.bounds = bounds
        self.q1 = q1
        self.r_flat = r_flat
        self.sq_t = sq_t
        self.C = C
        self.Xdd = Xdd
        self.Xdd_rev = Xdd_rev
        self.eye_q = eye_q
        self.noise_model = noise_model
        self.q_base_idx = q_base_idx
        self.mean_fn = mean_fn
        self.mean_param_start = mean_param_start
        self.prior_fn = prior_fn

    def _width_fxn(self, x, c, k, a):
        z1 = torch.clamp(-k * (x - a), min=-100.0, max=25.0)
        z2 = torch.clamp(-k * (x + a), min=-100.0, max=25.0)
        return -c / (1.0 + torch.exp(z1)) + c / (1.0 + torch.exp(z2))

    def _log_lik(self, theta_raw_val):
        device = torch.device(self.device_str)
        bounds = self.bounds
        q1 = self.q1
        r_flat = self.r_flat
        sq_t = self.sq_t
        C = self.C

        theta  = (bounds[:, 1] - bounds[:, 0]) * torch.sigmoid(theta_raw_val) + bounds[:, 0]
        ell, c, k, a = theta[0], theta[1], theta[2], theta[3]
        q_base = theta[self.q_base_idx]

        noise = theta[4] ** 2 * torch.ones(len(q1), dtype=q1.dtype, device=device)
        if self.noise_model == 'quadratic':
            noise = noise + theta[5] ** 2 * q1 ** 2

        sigma     = self._width_fxn(r_flat,  c, k, a)
        sigma_rev = self._width_fxn(-r_flat, c, k, a)
        Krr = f_kernel(self.Xdd, ell, sigma, sigma) + f_kernel(self.Xdd_rev, ell, sigma_rev, sigma)

        Kqq = C @ Krr @ C.T + torch.diag(noise)
        Kqq = (Kqq + Kqq.T) * 0.5 + self.eye_q * 1e-6

        mean_q_val = C @ real_space_mean(
            r_flat, theta, self.mean_fn, self.mean_param_start)

        sq_shifted = (sq_t.reshape(-1) - q_base - mean_q_val).reshape(-1, 1)
        L      = torch.linalg.cholesky(Kqq)
        logdet = 2.0 * torch.sum(torch.log(torch.diag(L)))
        alpha  = torch.cholesky_solve(sq_shifted, L)
        n      = len(q1)
        neg_log_lik = (0.5 * (sq_shifted.reshape(1, -1) @ alpha).squeeze()
                       + 0.5 * logdet
                       + n / 2.0 * np.log(2.0 * np.pi))
        return -neg_log_lik

    def __call__(self):
        import pyro
        import pyro.distributions as dist
        device = torch.device(self.device_str)
        theta_raw = pyro.sample(
            "theta_raw",
            dist.Normal(
                torch.zeros(self.P, dtype=torch.float64, device=device),
                3.0 * torch.ones(self.P, dtype=torch.float64, device=device),
            ).to_event(1),
        )
        if self.prior_fn == "uniform":
            # Replace the Normal sampling density with the logit Jacobian for
            # a uniform prior on the bounded physical parameters.
            normal_log_prob = dist.Normal(0.0, 3.0).log_prob(theta_raw).sum()
            log_jacobian = (torch.nn.functional.logsigmoid(theta_raw)
                            + torch.nn.functional.logsigmoid(-theta_raw)).sum()
            pyro.factor("prior_correction", log_jacobian - normal_log_prob)
        pyro.factor("log_lik", self._log_lik(theta_raw))


def nuts_sample(gp, r_grid, q_train, sq_train,
                num_samples=500, num_warmup=200,
                num_chains=1, seed=None):
    """
    Run NUTS to draw posterior samples of theta_raw.

    The log posterior is:
        log p(theta_raw | data) ∝ log p(F | theta_raw) + log p(theta_raw)
    where log p(F | theta_raw) is the GP marginal likelihood computed
    functionally from the sampled theta_raw (never touches gp.theta_raw).
    The prior follows gp.prior_fn: Gaussian N(0,9) on raw parameters,
    or uniform on bounded physical parameters with the logit Jacobian.

    Parameters
    ----------
    gp           : trained GP instance (used for bounds, noise_model, K, nearestPD)
    r_grid       : (Nr,1) integration grid
    q_train      : (Nq,1) training q-values
    sq_train     : (Nq,1) training F(q) values
    num_samples  : posterior draws after warmup
    num_warmup   : NUTS warmup / adaptation steps
    num_chains   : number of independent chains (>1 enables R-hat diagnostics)
    seed         : optional integer for reproducibility

    Returns
    -------
    mcmc         : Pyro MCMC object (call mcmc.summary() for R-hat / ESS)
    samples      : (num_samples, P) tensor of theta_raw draws (on CPU)
    """
    try:
        import pyro
        from pyro.infer import MCMC, NUTS
    except ImportError:
        raise ImportError("NUTS requires pyro-ppl: pip install pyro-ppl")

    if seed is not None:
        pyro.set_rng_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move all fixed tensors to device once
    r_g = r_grid.to(device=device, dtype=torch.float64)
    q_t = q_train.to(device=device, dtype=torch.float64)
    sq_t = sq_train.to(device=device, dtype=torch.float64)
    bounds = gp.theta_bounds.detach().to(device)

    r1 = r_g.reshape(-1)
    q1 = q_t.reshape(-1)
    dr = r1[1] - r1[0]
    weights = torch.ones_like(r1) * dr
    weights[0] *= 0.5
    weights[-1] *= 0.5
    C = torch.cos(torch.outer(q1, r1)) * weights
    Xdd     = custom_cdist(r_g, r_g)
    Xdd_rev = custom_cdist(-r_g, r_g)
    r_flat  = r_g.T[0]
    eye_q   = torch.eye(len(q1), dtype=torch.float64, device=device)

    model = _NUTSModel(
        P=gp.theta_raw.numel(),
        device_str=str(device),
        bounds=bounds,
        q1=q1,
        r_flat=r_flat,
        sq_t=sq_t,
        C=C,
        Xdd=Xdd,
        Xdd_rev=Xdd_rev,
        eye_q=eye_q,
        noise_model=gp.noise_model,
        q_base_idx=gp._q_base_idx,
        mean_fn=gp.mean_fn,
        mean_param_start=gp._mean_param_start_idx,
        prior_fn=gp.prior_fn,
    )

    mp_ctx = "spawn" if num_chains > 1 else None
    kernel = NUTS(model, adapt_step_size=True, jit_compile=True)
    mcmc = MCMC(kernel, num_samples=num_samples, warmup_steps=num_warmup,
                num_chains=num_chains, mp_context=mp_ctx)
    mcmc.run()
    return mcmc, mcmc.get_samples()["theta_raw"].cpu()  # (S, P)


def nuts_predict(gp, r_grid, q_train, sq_train, q_infer, r_infer,
                 num_samples=500, num_warmup=200, num_chains=1, seed=None):
    """
    NUTS-marginalised posterior in q and r space.

    Drop-in replacement for laplace_predict — same 4-tuple return signature.
    Runs nuts_sample internally and prints MCMC diagnostics (R-hat, ESS).
    Predictive covariance uses law of total variance over NUTS samples.

    Parameters
    ----------
    gp           : trained GP instance
    r_grid       : (Nr,1) integration grid used during training
    q_train      : (Nq,1) training q-values
    sq_train     : (Nq,1) training F(q) values
    q_infer      : (Nq*,1) inference q-grid
    r_infer      : (Nr*,1) inference r-grid
    num_samples  : posterior draws after warmup
    num_warmup   : NUTS warmup / adaptation steps
    num_chains   : number of independent chains
    seed         : optional integer for reproducibility

    Returns
    -------
    mu_q  : (Nq*,1)   posterior mean in q-space
    cov_q : (Nq*,Nq*) posterior covariance in q-space
    mu_r  : (Nr*,1)   posterior mean in r-space
    cov_r : (Nr*,Nr*) posterior covariance in r-space
    """
    mcmc, theta_samples = nuts_sample(gp, r_grid, q_train, sq_train,
                                      num_samples=num_samples,
                                      num_warmup=num_warmup,
                                      num_chains=num_chains,
                                      seed=seed)
    print("\n── NUTS diagnostics ──")
    mcmc.summary()

    theta_map = gp.theta_raw.detach().clone()
    q_means, q_covs, r_means, r_covs = [], [], [], []

    with torch.no_grad():
        for theta_s in theta_samples:
            gp.theta_raw.data.copy_(theta_s)
            gp.compute_params_from_raw()

            mu_q, cov_q = gp.predict_sq_trapz(r_grid, q_infer, q_train, sq_train,
                                               adjust=False)
            mu_r, cov_r = gp.predict_ed_trapz(r_grid, r_infer, q_train, sq_train,
                                               adjust=False)
            q_means.append(mu_q.reshape(-1))
            q_covs.append(cov_q)
            r_means.append(mu_r.reshape(-1))
            r_covs.append(cov_r)

        # Restore MAP parameters
        gp.theta_raw.data.copy_(theta_map)
        gp.compute_params_from_raw()

    def _mixture(means, covs):
        mu_stack = torch.stack(means)                              # (S, N)
        cov_stack = torch.stack(covs)                             # (S, N, N)
        mu_mix = mu_stack.mean(0)                                 # (N,)
        e_cov = cov_stack.mean(0)                                 # E[Σ_i]
        resid = mu_stack - mu_mix.unsqueeze(0)                    # (S, N)
        var_mu = (resid.unsqueeze(-1) * resid.unsqueeze(-2)).mean(0)  # (N, N)
        cov_mix = e_cov + var_mu
        return mu_mix.reshape(-1, 1), gp.nearestPD((cov_mix + cov_mix.T) / 2)

    mu_q_mix, cov_q_mix = _mixture(q_means, q_covs)
    mu_r_mix, cov_r_mix = _mixture(r_means, r_covs)
    return mu_q_mix, cov_q_mix, mu_r_mix, cov_r_mix


def laplace_covariance(gp, r_grid, q_train, sq_train):
    """
    Compute Laplace approximation covariance of theta_raw at MAP.
    Returns H^{-1} where H = d²(NEG_LMLH)/d(theta_raw)²

    Call after training (gp at MAP). Requires grad on gp.theta_raw.
    """
    loss = gp.NEG_LMLH_Trapz(r_grid, q_train, sq_train)
    grad = torch.autograd.grad(loss, gp.theta_raw, create_graph=True)[0].view(-1)
    P = grad.numel()

    H = torch.zeros(P, P, dtype=gp.theta_raw.dtype)
    for i in range(P):
        row = torch.autograd.grad(grad[i], gp.theta_raw,
                                  retain_graph=(i < P - 1))[0].view(-1)
        H[i] = row.detach()

    H = (H + H.T) / 2                          # symmetrize numerical noise
    H_pd = gp.nearestPD(H)
    L = torch.linalg.cholesky(H_pd)
    cov = torch.cholesky_inverse(L)
    return gp.nearestPD((cov + cov.T) / 2)


def laplace_predict(gp, r_grid, q_train, sq_train,
                    q_infer, r_infer,
                    num_samples=50, seed=None):
    """
    Laplace-marginalised posterior in q and r.

    Samples theta_raw ~ N(theta_raw_MAP, H^{-1}), runs predict for each,
    returns mixture mean and covariance via the law of total variance:

        Σ_total = E_θ[Σ_post(θ)] + Var_θ[μ_post(θ)]

    Returns: mu_q (Nq,1), cov_q (Nq,Nq), mu_r (Nr,1), cov_r (Nr,Nr)
    Drop-in replacement for predict_sq_trapz / predict_ed_trapz.
    """
    cov_laplace = laplace_covariance(gp, r_grid, q_train, sq_train)
    theta_map = gp.theta_raw.detach().clone()

    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)
    eps = torch.randn(num_samples, theta_map.numel(),
                      generator=gen, dtype=theta_map.dtype)
    L_cov = torch.linalg.cholesky(cov_laplace)
    theta_samples = theta_map.unsqueeze(0) + eps @ L_cov.T  # (S, P)

    q_means, q_covs = [], []
    r_means, r_covs = [], []

    with torch.no_grad():
        for theta_s in theta_samples:
            gp.theta_raw.copy_(theta_s)
            gp.compute_params_from_raw()

            # adjust=False: skip nearestPD on intermediate kernel matrices so the
            # Schur complement post_cov = Kii - Kid Kdd^{-1} Kid^T is not collapsed
            # by the PD projection.  nearestPD is applied to the mixture at the end.
            mu_q, cov_q = gp.predict_sq_trapz(r_grid, q_infer, q_train, sq_train,
                                               adjust=False)
            mu_r, cov_r = gp.predict_ed_trapz(r_grid, r_infer, q_train, sq_train,
                                               adjust=False)

            q_means.append(mu_q.reshape(-1))
            q_covs.append(cov_q)
            r_means.append(mu_r.reshape(-1))
            r_covs.append(cov_r)

        # Restore MAP params
        gp.theta_raw.copy_(theta_map)
        gp.compute_params_from_raw()

    def mixture(means, covs):
        mu_stack = torch.stack(means)                                        # (S, N)
        cov_stack = torch.stack(covs)                                        # (S, N, N)
        mu_mix = mu_stack.mean(0)                                            # (N,)
        e_cov = cov_stack.mean(0)                                            # E[Σ_i]
        resid = mu_stack - mu_mix.unsqueeze(0)                               # (S, N)
        var_mu = (resid.unsqueeze(-1) * resid.unsqueeze(-2)).mean(0)         # (N, N)
        cov_mix = e_cov + var_mu
        return mu_mix.reshape(-1, 1), gp.nearestPD((cov_mix + cov_mix.T) / 2)

    mu_q_mix, cov_q_mix = mixture(q_means, q_covs)
    mu_r_mix, cov_r_mix = mixture(r_means, r_covs)
    return mu_q_mix, cov_q_mix, mu_r_mix, cov_r_mix
