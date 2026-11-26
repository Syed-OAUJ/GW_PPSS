"""
waveform.py -- symbolic PN phasing (translated from AISS_3p5NonSpinWaveform_AlsoKappa.nb)
and helpers that turn it into fast numpy functions.

Single source of truth for the symbols and the waveform: fisher.py and bayes.py
both build on `phase_expr`, so they can never disagree about the model.

Parameter vector:  THETA = (t_c, phi_c, M_c, eta, kappa_s)
Fixed inputs:      chi1, chi2, kappa_a
(Dropped from the old Analytical.py: Section 1 `psi`/`h`/`conjh1`, which only
summed alpha0..alpha3 and was never used by the Fisher/MCMC code.)
"""
from functools import lru_cache

import numpy as np
import sympy as sp
from sympy import Rational, log, pi, sqrt

import config as cfg

# ----------------------------------------------------------------------------
# Symbols (one shared set for the whole project)
# ----------------------------------------------------------------------------
f, M_c, eta = sp.symbols("f M_c eta", positive=True)
t_c, phi_c, kappa_s, kappa_a, chi1, chi2 = sp.symbols(
    "t_c phi_c kappa_s kappa_a chi1 chi2", real=True
)

THETA = [t_c, phi_c, M_c, eta, kappa_s]
# argument order of every lambdified function below
ARGS = (f, t_c, phi_c, M_c, eta, kappa_s, chi1, chi2, kappa_a)


# ----------------------------------------------------------------------------
# PN phasing
# ----------------------------------------------------------------------------
def pn_phasing(f_, M_, eta_, chi1_, chi2_, kappas_, kappaa_, max_order=7):
    """
    Spin-aligned + quadrupole-monopole (kappa_s, kappa_a) phasing, truncated at
    alpha_<max_order> (max_order=3 -> up to 1.5PN, 7 -> 3.5PN).
    alpha1 is identically zero, so it never appears.
    """
    v = (pi * M_ * f_) ** Rational(1, 3)
    vlso = (pi * M_ * cfg.F_REF_LOG) ** Rational(1, 3)

    alpha0 = Rational(3) / (128 * eta_ * v ** 5)
    alpha2 = alpha0 * (Rational(20, 9) * (Rational(743, 336) + Rational(11, 4) * eta_))

    delta = sqrt(1 - 4 * eta_)
    ca = (chi1_ - chi2_) / 2           # chi_a
    cs = (chi1_ + chi2_) / 2           # chi_s
    kp = 2 * kappas_
    km = 2 * kappaa_

    alpha3 = alpha0 * (
        -16 * pi
        + Rational(113, 3) * delta * ca
        + (Rational(113, 3) - Rational(76, 3) * eta_) * cs
    )
    alpha4 = alpha0 * (
        Rational(15293365, 508032) + Rational(27145, 504) * eta_ + Rational(3085, 72) * eta_ ** 2
        + (-Rational(5, 8) + 100 * eta_ - 25 * kp + 50 * eta_ * kp) * ca ** 2
        + (-50 * km + 100 * eta_ * km) * ca * cs
        + (-Rational(5, 8) - Rational(195, 2) * eta_ - 25 * kp + 50 * eta_ * kp) * cs ** 2
        + delta * (
            -25 * km * ca ** 2
            + (-Rational(5, 4) - 50 * kp) * ca * cs
            - 25 * km * cs ** 2
        )
    )
    alpha5 = alpha0 * pi * (
        Rational(38645, 756)
        + Rational(38645, 252) * log(v / vlso)
        + Rational(5, 3) * eta_ * (1 + 3 * log(v / vlso))
    )
    alpha6 = alpha0 * (
        (Rational(11583231236531, 4694215680) - Rational(640, 3) * pi ** 2
         - Rational(6848, 21) * cfg.GAMMA_E)
        + eta_ * (
            Rational(-15335597827, 3048192) + Rational(2255, 12) * pi ** 2
            - Rational(1760, 3) * cfg.THETA_PN + Rational(12320, 9) * cfg.LAMBDA_PN
        )
        + Rational(76055, 1728) * eta_ ** 2
        - Rational(127825, 1296) * eta_ ** 3
        - Rational(6848, 21) * log(4 * v)
    )
    alpha7 = alpha0 * pi * (
        Rational(77096675, 254016) + Rational(1014115, 3024) * eta_
        - Rational(36868, 378) * eta_ ** 2
    )

    coeffs = {2: alpha2, 3: alpha3, 4: alpha4, 5: alpha5, 6: alpha6, 7: alpha7}
    higher = sum(a * v ** n for n, a in coeffs.items() if n <= max_order)
    return alpha0 + higher - pi / 4


def phase_expr(max_order=7):
    """Full Fourier-domain phase Psi(f) + 2 pi f t_c - phi_c, in terms of M_c, eta."""
    M_total = M_c / eta ** Rational(3, 5)
    psi = pn_phasing(f, M_total, eta, chi1, chi2, kappa_s, kappa_a, max_order)
    return psi + 2 * pi * f * t_c - phi_c


@lru_cache(maxsize=None)
def waveform_function(max_order=7):
    """Numpy function h(f, t_c, phi_c, M_c, eta, kappa_s, chi1, chi2, kappa_a), unit amplitude."""
    h = f ** Rational(-7, 6) * sp.exp(sp.I * phase_expr(max_order))
    return sp.lambdify(ARGS, h, modules="numpy")


# ----------------------------------------------------------------------------
# Binary / frequency-grid helpers
# ----------------------------------------------------------------------------
def masses_to_system(m1_msun, m2_msun):
    """(m1, m2) in Msun -> (M_total [s], eta, M_chirp [s])."""
    M = (m1_msun + m2_msun) * cfg.MSUN_SEC
    eta_val = m1_msun * m2_msun / (m1_msun + m2_msun) ** 2
    return M, eta_val, eta_val ** 0.6 * M


def total_mass_q_to_system(M_tot_msun, q):
    """(M_total [Msun], q = m1/m2 >= 1) -> (M_total [s], eta, M_chirp [s])."""
    eta_val = q / (1.0 + q) ** 2
    M = M_tot_msun * cfg.MSUN_SEC
    return M, eta_val, eta_val ** 0.6 * M


def f_lso(M_sec):
    """Schwarzschild last-stable-orbit GW frequency [Hz], M in seconds."""
    return 1.0 / (6.0 ** 1.5 * np.pi * M_sec)


def make_freq_grid(M_sec, fs=cfg.F_LOW, f_upper_cap=None, n=cfg.N_FREQS_FISHER):
    """Log-spaced grid from fs to f_lso (optionally capped)."""
    f_max = f_lso(M_sec)
    if f_upper_cap is not None:
        f_max = min(f_max, f_upper_cap)
    if f_max <= fs:
        raise ValueError(f"f_lso ({f_max:.1f} Hz) <= fs ({fs} Hz): system merges below band.")
    return np.logspace(np.log10(fs), np.log10(f_max), n)
