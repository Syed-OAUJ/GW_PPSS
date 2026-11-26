"""
fisher.py -- Fisher-matrix forecasts for theta = (t_c, phi_c, M_c, eta, kappa_s).

For h = A f^(-7/6) exp(i Phi(f)) with a real parameter-dependent phase,
    Gamma_ij = 4 A^2 int f^(-7/3) dPhi/dtheta_i dPhi/dtheta_j / S_h(f) df,
which is identical to the old 2*(conj(g_i) g_j + g_i conj(g_j))/S_h form but needs
no symbolic conjugation. Symbolic derivatives are lambdified once per PN order
and cached, so sweeps (Fig. 2) are fast.
"""
from functools import lru_cache

import numpy as np
import pandas as pd
import sympy as sp
from scipy.integrate import trapezoid

import config as cfg
import psd
import waveform as wf


@lru_cache(maxsize=None)
def _dphase_functions(max_order):
    phase = wf.phase_expr(max_order)
    return [sp.lambdify(wf.ARGS, sp.diff(phase, p), modules="numpy") for p in wf.THETA]


def _on_grid(fn, freqs, params):
    """Evaluate on the grid, broadcasting constants (e.g. an exact 0 for dPhi/dkappa_s below 2PN)."""
    val = np.asarray(fn(freqs, *params), dtype=float)
    return np.broadcast_to(val, freqs.shape)


def fisher_matrix(Mc, eta, kappa_s, kappa_a, chi1, chi2,
                  t_c=cfg.T_C, phi_c=cfg.PHI_C,
                  max_order=cfg.MAX_PN_ORDER, fs=cfg.F_LOW, f_upper_cap=None,
                  n_freqs=cfg.N_FREQS_FISHER, rho=cfg.SNR_FIDUCIAL):
    """5x5 Fisher matrix at SNR = rho. Masses in seconds (G = c = 1)."""
    if abs(eta - 0.25) < 1e-6 and chi1 != chi2:
        raise ValueError("eta is numerically 0.25 while chi1 != chi2: d(delta)/d(eta) diverges. "
                         "Use slightly unequal masses.")

    freqs = wf.make_freq_grid(Mc / eta ** 0.6, fs=fs, f_upper_cap=f_upper_cap, n=n_freqs)
    params = (t_c, phi_c, Mc, eta, kappa_s, chi1, chi2, kappa_a)
    dphi = np.array([_on_grid(fn, freqs, params) for fn in _dphase_functions(max_order)])

    weight = 4.0 * freqs ** (-7 / 3) / psd.sh_aligo_new(freqs)
    unit = trapezoid(weight * dphi[:, None, :] * dphi[None, :, :], freqs, axis=-1)

    amplitude = psd.amplitude_for_snr(psd.sh_aligo_new, freqs, rho)
    return amplitude ** 2 * unit


def covariance_from_fisher(fisher):
    """
    Inverse Fisher matrix. If kappa_s carries no information (max_order < 4, where
    kappa first enters), invert the remaining block and return NaN for kappa_s.
    """
    k = cfg.KAPPA_S_INDEX
    if not np.allclose(fisher[k, :], 0.0):
        return np.linalg.inv(fisher)

    keep = [i for i in range(fisher.shape[0]) if i != k]
    cov = np.full(fisher.shape, np.nan)
    cov[np.ix_(keep, keep)] = np.linalg.inv(fisher[np.ix_(keep, keep)])
    return cov


def fisher_covariance(*args, verbose=False, **kwargs):
    """Fisher matrix -> covariance. Same arguments as `fisher_matrix`."""
    fisher = fisher_matrix(*args, **kwargs)
    cov = covariance_from_fisher(fisher)
    if verbose:
        print("\nFisher matrix\n", pd.DataFrame(fisher).round(5))
        print("\nCovariance\n", pd.DataFrame(cov).round(5))
        if np.isnan(cov[cfg.KAPPA_S_INDEX, cfg.KAPPA_S_INDEX]):
            print("\nkappa_s has zero Fisher information at this PN order -> Delta kappa_s = NaN")
    return cov


def error_table_by_pn_order(Mc, eta, kappa_s, kappa_a, chi1, chi2,
                            fs=cfg.F_LOW, f_upper_cap=None, verbose=True):
    """Table I of the paper: 1-sigma errors versus PN order (1PN ... 3.5PN)."""
    rows, labels = [], []
    for order in range(2, 8):
        cov = fisher_covariance(Mc, eta, kappa_s, kappa_a, chi1, chi2,
                                max_order=order, fs=fs, f_upper_cap=f_upper_cap)
        s = np.sqrt(np.diag(cov))
        rows.append([s[0] * 1000, s[1], 100 * s[2] / Mc, 100 * s[3] / eta, s[4]])
        labels.append(cfg.PN_LABELS[order])

    df = pd.DataFrame(rows, index=labels,
                      columns=[r"$\Delta t_c$ (ms)", r"$\Delta\phi_c$",
                               r"$\Delta M_c/M_c$ (%)", r"$\Delta\eta/\eta$ (%)",
                               r"$\Delta\kappa_s$"])
    if verbose:
        print(df.round(4))
    return df


@lru_cache(maxsize=None)
def delta_kappa_s(M_tot_msun, q, chi1, chi2):
    """1-sigma error on kappa_s for a BBH fiducial (kappa_s=1, kappa_a=0). NaN if it fails."""
    _, eta, Mc = wf.total_mass_q_to_system(M_tot_msun, q)
    try:
        cov = fisher_covariance(Mc, eta, cfg.KAPPA_S_BH, cfg.KAPPA_A_BH, chi1, chi2)
        return float(np.sqrt(cov[cfg.KAPPA_S_INDEX, cfg.KAPPA_S_INDEX]))
    except (ValueError, np.linalg.LinAlgError) as err:
        print(f"  skipped M={M_tot_msun}, q={q}, chi=({chi1},{chi2}): {err}")
        return float("nan")


def sweep_delta_kappa_s(masses, q, chi1, chi2):
    return np.array([delta_kappa_s(float(M), float(q), float(chi1), float(chi2)) for M in masses])
