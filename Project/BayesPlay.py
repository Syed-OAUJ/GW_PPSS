import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import sympy as sp
from scipy.integrate import trapezoid

from Analytical import AISSKappa
from Plotting import Sh_new_func
from FisherKappaPlay import (
    MSUN_SEC, make_freq_grid, calc_amplitude,
    CreateFisherMatrice, mass_kappa_from_rows,
)

##############################################################################
# FIXED (non-sampled) parameters -- same convention as FisherKappaPlay:
# kappa2, chi1, chi2 stay fixed; theta = (t_c, phi_c, M_c, eta, kappa1)
##############################################################################

chi1_val, chi2_val = 0.9, 0.8

vals = mass_kappa_from_rows()
m1, kappa1_val = vals["mass1"], vals["kappa1"]
m2, kappa2_val = vals["mass2"], vals["kappa2"]

kappa_s_val = (kappa1_val + kappa2_val) / 2
kappa_a_val = (kappa1_val - kappa2_val) / 2   # fixed, not sampled

M_val = (m1 + m2) * MSUN_SEC
eta_val = m1 * m2 / (m1 + m2) ** 2
Mc_val = eta_val ** 0.6 * M_val
t_c_val, phi_c_val = 0.0, 0.0

theta_true = np.array([t_c_val, phi_c_val, Mc_val, eta_val, kappa_s_val])
PARAM_LABELS = [r"$t_c$", r"$\phi_c$", r"$M_c$", r"$\eta$", r"$\kappa_s$"]

##############################################################################
# SYMBOLIC WAVEFORM
##############################################################################

f_, t_c_, phi_c_, M_c_, eta_, kappa_s_, chi1_, chi2_ = sp.symbols(
    "f t_c phi_c M_c eta kappa_s chi1 chi2"
)

M_total_ = M_c_ / eta_ ** sp.Rational(3, 5)

# kappa_a_val is baked in as a plain float -- no symbol, nothing to substitute later
psi_sym = AISSKappa(f_, M_total_, eta_, chi1_, chi2_, kappa_s_, kappa_a_val, max_order=7)
full_phase_sym = psi_sym + 2 * sp.pi * f_ * t_c_ - phi_c_
h_sym = f_ ** sp.Rational(-7, 6) * sp.exp(sp.I * full_phase_sym)

h_sym_fixed = h_sym.subs({chi1_: chi1_val, chi2_: chi2_val})

waveform_func = sp.lambdify(
    (f_, t_c_, phi_c_, M_c_, eta_, kappa_s_), h_sym_fixed, modules="numpy"
)
##############################################################################
# FREQUENCY GRID, PSD, AMPLITUDE (SNR=10 fiducial), "DATA"
##############################################################################

freqs = make_freq_grid(M_val, fs=20.0, n=2000)
Sh_vals = Sh_new_func(freqs)
amplitude = calc_amplitude(Sh_new_func, freqs, rho=10.0)


def waveform(theta):
    t_c, phi_c, M_c, eta, kappa1 = theta
    return amplitude * waveform_func(freqs, t_c, phi_c, M_c, eta, kappa1)


def generate_data(theta_true, add_noise=False, seed=42):
    h_true = waveform(theta_true)
    if not add_noise:
        return h_true  # zero-noise injection: posterior should recover theta_true exactly
    rng = np.random.default_rng(seed)
    df_local = np.gradient(freqs)  # local bin width -- grid is log-spaced, not uniform
    noise_std = np.sqrt(Sh_vals / (4 * df_local))
    noise = (rng.normal(size=len(freqs)) + 1j * rng.normal(size=len(freqs))) * noise_std
    return h_true + noise


data = generate_data(theta_true, add_noise=False)

##############################################################################
# LIKELIHOOD / PRIOR / POSTERIOR
##############################################################################


def log_likelihood(theta):
    diff = data - waveform(theta)
    # matches the (a|b) convention in CreateFisherMatrice's Fisher elements:
    # (a|b) = 2*int[conj(a)b + a*conj(b)]/Sh df = 4*int Re(conj(a)b)/Sh df
    integrand = 4.0 * np.abs(diff) ** 2 / Sh_vals
    return -0.5 * trapezoid(integrand, freqs)  # trapezoid handles the non-uniform log grid


def log_prior(theta):
    t_c, phi_c, M_c, eta, kappa_s = theta
    if not (-0.1 < t_c < 0.1):
        return -np.inf
    if not (-np.pi <= phi_c < np.pi):
        return -np.inf
    if not (0.5 * Mc_val < M_c < 1.5 * Mc_val):
        return -np.inf
    if not (0.05 < eta <= 0.25):
        return -np.inf
    if not (0.0 < kappa_s < 10.0):
        return -np.inf
    return 0.0


def log_posterior(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


if __name__ == "__main__":

    ##############################################################################
    # SANITY CHECK -- catch broken likelihoods before the expensive run
    ##############################################################################

    lp0 = log_posterior(theta_true)
    print("log_posterior at true value:", lp0, " -- should be finite (0.0 in the zero-noise case)")
    assert np.isfinite(lp0), "log_posterior is not finite at the true parameters -- fix before running MCMC"

    ndim, nwalkers, nsteps = 5, 32, 3000

    covariance = CreateFisherMatrice(
        t_c_val=t_c_val, phi_c_val=phi_c_val,
        M_c_val=Mc_val, eta_val=eta_val,
        kappa_s_val=kappa_s_val, kappa_a_val=kappa_a_val,
        chi1_val=chi1_val, chi2_val=chi2_val,
        max_order=7,
    )

    if np.all(np.isfinite(covariance)) and np.all(np.diag(covariance) > 0):
        scale = np.sqrt(np.diag(covariance))
    else:
        print("WARNING: Fisher covariance has non-finite/non-positive diagonal entries "
              "-- falling back to a rough manual scale")
        scale = np.array([1e-4, 0.1, 0.05 * Mc_val, 0.02, 0.2])

    rng = np.random.default_rng(7)
    pos = theta_true + 0.5 * scale * rng.standard_normal((nwalkers, ndim))
    pos[:, 3] = np.clip(pos[:, 3], 0.05 + 1e-6, 0.25 - 1e-6)
    pos[:, 4] = np.clip(pos[:, 4], 1e-6, 10.0 - 1e-6)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)
    sampler.run_mcmc(pos, nsteps, progress=True)
    print(f"Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.2f}")

    flat_samples = sampler.get_chain(discard=500, thin=10, flat=True)
    np.save("mcmc_samples.npy", flat_samples)   # <-- new: so CompareFisherBayes.py can reuse it

    fig = corner.corner(
        flat_samples, labels=PARAM_LABELS, truths=list(theta_true),
        quantiles=[0.16, 0.5, 0.84], show_titles=True
    )
    fig.savefig("kappa1_corner.png", dpi=150)
    plt.show()