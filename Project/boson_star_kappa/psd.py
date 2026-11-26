"""
psd.py -- detector noise curves and PSD-weighted integrals.

All curves are plain numpy functions of frequency (no sympy needed).
Importing this module has no side effects (the old Plotting.py read a CSV and
printed an SNR at import time).
"""
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

import config as cfg


# ----------------------------------------------------------------------------
# Analytical fits
# ----------------------------------------------------------------------------
def sh_aligo_new(freqs, f0=None, S0=None):
    """Adv. LIGO fit, Eq. (3.7a): S0 [x^-4.14 - 5x^-2 + 111(1 - x^2 + x^4/2)/(1 + x^2/2)]."""
    p = cfg.ALIGO_NEW_FIT
    f0 = p["f0"] if f0 is None else f0
    S0 = p["S0"] if S0 is None else S0
    x = np.asarray(freqs, dtype=float) / f0
    return S0 * (x ** -4.14 - 5 * x ** -2 + 111 * (1 - x ** 2 + x ** 4 / 2) / (1 + x ** 2 / 2))


def sh_aligo_old(freqs, f0=None, S0=None):
    """Older fit used in Refs. [6,22,25,46]: S0 [x^-4 + 2 + 2x^2]."""
    p = cfg.ALIGO_OLD_FIT
    f0 = p["f0"] if f0 is None else f0
    S0 = p["S0"] if S0 is None else S0
    x = np.asarray(freqs, dtype=float) / f0
    return S0 * (x ** -4 + 2 + 2 * x ** 2)


def sh_ajith(freqs):
    """Advanced LIGO analytical PSD fit (Ajith)."""
    x = np.asarray(freqs, dtype=float) / 245.4
    return 1e-48 * (0.0152 * x ** -4 + 0.2935 * x ** 2.25 + 2.7951 * x ** 1.5
                    - 6.5080 * x ** 0.75 + 17.7622)


def sh_einstein_telescope(freqs):
    """Einstein Telescope analytical PSD fit."""
    x = np.asarray(freqs, dtype=float) / 100.0
    return 1e-50 * (2.39e-27 * x ** -15.64 + 0.349 * x ** -2.145
                    + 1.76 * x ** -0.12 + 0.409 * x ** 1.10)


# ----------------------------------------------------------------------------
# Numerical PSD + SNR integrals
# ----------------------------------------------------------------------------
def load_numerical_psd(path=cfg.NUMERICAL_PSD_FILE):
    return pd.read_csv(path, sep=r"\s+", header=None, names=["freq", "psd"])


def snr_density(freq, psd):
    """Integrand f^(-7/3) / S_h(f) of the SNR^2 for a unit-amplitude inspiral."""
    return np.asarray(freq) ** (-7 / 3) / np.asarray(psd)


def amplitude_for_snr(sh_func, freqs, rho):
    """Amplitude A such that a signal A * f^(-7/6) e^{i phi} has SNR = rho."""
    integral = trapezoid(freqs ** (-7 / 3) / sh_func(freqs), freqs)
    return rho / (2.0 * np.sqrt(integral))


# ----------------------------------------------------------------------------
# Optional PyCBC comparison (imports are lazy so the rest works without PyCBC)
# ----------------------------------------------------------------------------
def measured_psds_gw150914(ifos=("H1", "L1")):
    from pycbc.catalog import Merger
    from pycbc.filter import highpass_fir
    from pycbc.psd import welch, interpolate

    out = {}
    for ifo in ifos:
        strain = highpass_fir(Merger("GW150914").strain(ifo), 15, 8)
        out[ifo] = interpolate(welch(strain), 1.0 / strain.duration)
    return out


def synthetic_heavy_bh_strain_squared():
    """|h+|^2 + |hx|^2 of an IMRPhenomD waveform (note: a strain, not a PSD)."""
    from pycbc.waveform import get_fd_waveform

    hp, hc = get_fd_waveform(
        approximant="IMRPhenomD",
        mass1=cfg.SYNTH_MASS1, mass2=cfg.SYNTH_MASS2,
        delta_f=cfg.SYNTH_DELTA_F, f_lower=cfg.SYNTH_F_LOWER, f_final=cfg.SYNTH_F_FINAL,
    )
    h_net = abs(hp) ** 2 + abs(hc) ** 2
    return h_net.sample_frequencies.numpy(), h_net
