"""
bayes.py -- Bayesian parameter estimation (emcee) for theta = (t_c, phi_c, M_c, eta, kappa_s).

`BayesModel` bundles the injection, PSD, likelihood and prior, so nothing runs at
import time (the old BayesPlay.py built the model and "data" on import).
Likelihood convention matches fisher.py: (a|b) = 4 Re int conj(a) b / S_h df.
"""
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

import config as cfg
import psd
import waveform as wf


class BayesModel:
    def __init__(self, Mc, eta, kappa_s, kappa_a, chi1, chi2,
                 t_c=cfg.T_C, phi_c=cfg.PHI_C,
                 max_order=cfg.MAX_PN_ORDER, fs=cfg.F_LOW, n_freqs=cfg.N_FREQS_MCMC,
                 rho=cfg.SNR_FIDUCIAL, add_noise=cfg.ADD_NOISE, noise_seed=cfg.NOISE_SEED):
        self.Mc_true = Mc
        self.kappa_a, self.chi1, self.chi2 = kappa_a, chi1, chi2
        self.theta_true = np.array([t_c, phi_c, Mc, eta, kappa_s])

        self.freqs = wf.make_freq_grid(Mc / eta ** 0.6, fs=fs, n=n_freqs)
        self.sh = psd.sh_aligo_new(self.freqs)
        self.amplitude = psd.amplitude_for_snr(psd.sh_aligo_new, self.freqs, rho)
        self._h = wf.waveform_function(max_order)
        self.data = self._make_data(add_noise, noise_seed)

    def waveform(self, theta):
        t_c, phi_c, Mc, eta, kappa_s = theta
        return self.amplitude * self._h(self.freqs, t_c, phi_c, Mc, eta, kappa_s,
                                        self.chi1, self.chi2, self.kappa_a)

    def _make_data(self, add_noise, seed):
        h_true = self.waveform(self.theta_true)
        if not add_noise:
            return h_true            # zero-noise injection
        rng = np.random.default_rng(seed)
        df_local = np.gradient(self.freqs)     # log-spaced grid -> local bin width
        std = np.sqrt(self.sh / (4 * df_local))
        return h_true + (rng.normal(size=len(self.freqs)) + 1j * rng.normal(size=len(self.freqs))) * std

    def log_likelihood(self, theta):
        diff = self.data - self.waveform(theta)
        return -0.5 * trapezoid(4.0 * np.abs(diff) ** 2 / self.sh, self.freqs)

    def log_prior(self, theta):
        t_c, phi_c, Mc, eta, kappa_s = theta
        p = cfg.PRIOR
        lo, hi = p["Mc_fraction"]
        ok = (p["t_c"][0] < t_c < p["t_c"][1]
              and p["phi_c"][0] <= phi_c < p["phi_c"][1]
              and lo * self.Mc_true < Mc < hi * self.Mc_true
              and p["eta"][0] < eta <= p["eta"][1]
              and p["kappa_s"][0] < kappa_s < p["kappa_s"][1])
        return 0.0 if ok else -np.inf

    def log_posterior(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_likelihood(theta)
        return lp + ll if np.isfinite(ll) else -np.inf


def run_mcmc(model, fisher_cov=None, n_walkers=cfg.N_WALKERS, n_steps=cfg.N_STEPS,
             spread=cfg.INIT_SPREAD, seed=cfg.INIT_SEED, progress=True):
    """Run emcee, initialising walkers in a ball sized by the Fisher sigmas."""
    import emcee

    lp0 = model.log_posterior(model.theta_true)
    assert np.isfinite(lp0), "log_posterior is not finite at the true parameters"
    print(f"log_posterior at truth: {lp0} (0.0 expected for zero-noise injection)")

    if (fisher_cov is not None and np.all(np.isfinite(fisher_cov))
            and np.all(np.diag(fisher_cov) > 0)):
        scale = np.sqrt(np.diag(fisher_cov))
    else:
        print("WARNING: bad Fisher covariance -> using rough manual scale")
        fb = cfg.FALLBACK_SCALE
        scale = np.array([fb["t_c"], fb["phi_c"], fb["Mc_fraction"] * model.Mc_true,
                          fb["eta"], fb["kappa_s"]])

    ndim = len(model.theta_true)
    rng = np.random.default_rng(seed)
    pos = model.theta_true + spread * scale * rng.standard_normal((n_walkers, ndim))
    pos[:, 3] = np.clip(pos[:, 3], cfg.PRIOR["eta"][0] + 1e-6, cfg.PRIOR["eta"][1] - 1e-6)
    pos[:, 4] = np.clip(pos[:, 4], 1e-6, cfg.PRIOR["kappa_s"][1] - 1e-6)

    sampler = emcee.EnsembleSampler(n_walkers, ndim, model.log_posterior)
    sampler.run_mcmc(pos, n_steps, progress=progress)
    print(f"Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.2f}")
    return sampler


def flatten_chain(sampler, burn_in=cfg.BURN_IN, thin=cfg.THIN):
    return sampler.get_chain(discard=burn_in, thin=thin, flat=True)


def comparison_table(flat_samples, theta_true, fisher_sigma, labels):
    """Injected vs recovered (median, +-1 sigma from 16/84 percentiles) vs Fisher sigma."""
    med = np.median(flat_samples, axis=0)
    lo, hi = np.percentile(flat_samples, [16, 84], axis=0)
    return pd.DataFrame({
        "injected": theta_true, "recovered (median)": med,
        "recovered -1sigma": med - lo, "recovered +1sigma": hi - med,
        "Fisher sigma": fisher_sigma,
    }, index=[l.strip("$") for l in labels])
