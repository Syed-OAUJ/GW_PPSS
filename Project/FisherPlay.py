from Analytical import hAISSKappa, AISSKappa
from Plotting import Sh_new_numeric, Sh_new_func, f as f_plot

import sympy as sp
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

##############################################################################
# UNIT CONVENTION
##############################################################################
# All masses passed in (M_c_val, and M = M_c/eta**0.6) must be in SECONDS
# (G=c=1), not solar masses, since v=(pi*M*f)^(1/3) must be dimensionless.
MSUN_SEC = 4.925491025543576e-6  # G*M_sun/c^3, seconds


##############################################################################
# SYMBOLIC VARIABLES
##############################################################################

f_, t_c_, phi_c_, M_c_, eta_, kappa_a_, kappa_s_, chi1_, chi2_ = sp.symbols(
    "f t_c phi_c M_c eta kappa_a kappa_s chi1 chi2"
)


##############################################################################
# AMPLITUDE: unit-amplitude h -> physical A giving SNR = rho
##############################################################################

def calc_amplitude(Sh_func, freqs, rho=10.0):
    """rho^2 = 4 A^2 int f^(-7/3)/Sh(f) df  =>  A = rho / (2 sqrt(integral))"""
    integral = trapezoid(freqs ** (-7 / 3) / Sh_func(freqs), freqs)
    return rho / (2.0 * np.sqrt(integral))


##############################################################################
# LSO cutoff, Eq. (3.6) from AISS paper
##############################################################################

def f_lso(M_val):
    """Schwarzschild LSO frequency, M_val in seconds."""
    return 1.0 / (6.0 ** 1.5 * np.pi * M_val)


def make_freq_grid(M_val, fs=20.0, f_upper_cap=None, n=500):
    f_max = f_lso(M_val)
    if f_upper_cap is not None:
        f_max = min(f_max, f_upper_cap)
    if f_max <= fs:
        raise ValueError(
            f"f_lso ({f_max:.1f} Hz) <= fs ({fs} Hz): system merges below band."
        )
    return np.logspace(np.log10(fs), np.log10(f_max), n)


##############################################################################
# FISHER MATRIX
##############################################################################

def CreateFisherMatrice(
        t_c_val, phi_c_val, M_c_val, eta_val,
        kappa_a_val, kappa_s_val,
        chi1_val, chi2_val,
        max_order=7,
        fs=20.0,
        f_upper_cap=None,
        n_freqs=500,
        should_print_result=False
):
    """
    theta = (t_c, phi_c, M_c, eta). Builds h(f) = A f^(-7/6) exp(i full_phase)
    with full_phase = AISSKappa(f, M=M_c/eta^(3/5), eta, chi1, chi2,
    kappa_s, kappa_a) + 2*pi*f*t_c - phi_c, differentiates it directly
    (fixes the missing t_c/phi_c dependence and the M_c->M conversion that
    were lost before), builds Gamma_ab (Eq. 2.11), scales by the physical
    A^2 for SNR=10 (Eq. 2.8), and returns the covariance matrix.
    """

    theta = [t_c_, phi_c_, M_c_, eta_]

    M_total_ = M_c_ / eta_ ** sp.Rational(3, 5)

    psi = AISSKappa(
        f_, M_total_, eta_,
        chi1_, chi2_,
        kappa_s_, kappa_a_,
        max_order
    )
    full_phase = psi + 2 * sp.pi * f_ * t_c_ - phi_c_

    h = f_ ** sp.Rational(-7, 6) * sp.exp(sp.I * full_phase)

    grad = [sp.I * h * sp.diff(full_phase, par) for par in theta]

    Sh = Sh_new_numeric.subs(f_plot, f_)

    fisher = sp.zeros(len(theta))
    for i in range(len(theta)):
        for j in range(i, len(theta)):
            fisher[i, j] = (
                2 * (sp.conjugate(grad[i]) * grad[j] + grad[i] * sp.conjugate(grad[j]))
                / Sh
            )
            fisher[j, i] = fisher[i, j]

    fiducial = {
        t_c_: t_c_val, phi_c_: phi_c_val,
        M_c_: M_c_val, eta_: eta_val,
        chi1_: chi1_val, chi2_: chi2_val,
        kappa_s_: kappa_s_val, kappa_a_: kappa_a_val,
    }
    fisher = fisher.subs(fiducial)

    M_val = M_c_val / eta_val ** 0.6
    freqs = make_freq_grid(M_val, fs=fs, f_upper_cap=f_upper_cap, n=n_freqs)

    fisher_fun = sp.lambdify(f_, fisher, modules="numpy")
    raw = fisher_fun(freqs)

    fisher_vals = np.empty((4, 4, len(freqs)), dtype=np.complex128)
    for i in range(4):
        for j in range(4):
            fisher_vals[i, j, :] = raw[i][j]

    fisher_total_unit = np.asarray(
        trapezoid(fisher_vals.real, freqs, axis=2), dtype=float
    )

    amplitude = calc_amplitude(Sh_new_func, freqs)
    fisher_total = amplitude ** 2 * fisher_total_unit

    covariance = np.linalg.inv(fisher_total)

    if should_print_result:
        print("\nFisher matrix\n", pd.DataFrame(fisher_total).round(5))
        print("\nCovariance\n", pd.DataFrame(covariance).round(5))
        print("\nCheck Gamma @ Sigma = I\n", np.round(fisher_total @ covariance, 5))

    return covariance


##############################################################################
# TABLE I RECREATION
##############################################################################

_PN_LABELS = {2: "1PN", 3: "1.5PN", 4: "2PN", 5: "2.5PN", 6: "3PN", 7: "3.5PN"}


def TableIRecreate(
        t_c_val, phi_c_val, M_c_val, eta_val,
        chi1_val, chi2_val,
        kappa_s_val, kappa_a_val,
        fs=20.0, f_upper_cap=None,
        should_print_result=True
):
    rows, labels = [], []

    for order in range(2, 8):  # alpha_2 (1PN) ... alpha_7 (3.5PN)
        covariance = CreateFisherMatrice(
            t_c_val=t_c_val, phi_c_val=phi_c_val,
            M_c_val=M_c_val, eta_val=eta_val,
            chi1_val=chi1_val, chi2_val=chi2_val,
            kappa_s_val=kappa_s_val, kappa_a_val=kappa_a_val,
            max_order=order, fs=fs, f_upper_cap=f_upper_cap,
        )
        sigma = np.sqrt(np.diag(covariance))
        rows.append([
            sigma[0] * 1000,            # Delta t_c, ms
            sigma[1],                   # Delta phi_c, rad
            100 * sigma[2] / M_c_val,   # Delta M_c/M_c, %
            100 * sigma[3] / eta_val,   # Delta eta/eta, %
        ])
        labels.append(_PN_LABELS[order])

    df = pd.DataFrame(
        rows, index=labels,
        columns=[r"$\Delta t_c$ (ms)", r"$\Delta\phi_c$",
                 r"$\Delta M_c/M_c$ (%)", r"$\Delta\eta/\eta$ (%)"]
    )
    if should_print_result:
        print(df.round(4))
    return df


##############################################################################
# QUICK VALIDATION AGAINST TABLE I (non-spinning limit: chi1=chi2=0)
##############################################################################

def _validate_against_paper():
    systems = {"NS-NS": (1.4, 1.4), "NS-BH": (1.4, 10.0), "BH-BH": (10.0, 10.0)}

    # Table I, Advanced LIGO column: (dtc[ms], dphic, dM/M[%], deta/eta[%])
    paper_1PN = {
        "NS-NS": (0.3977, 0.9256, 0.0267, 4.656),
        "NS-BH": (0.5959, 1.261, 0.1420, 7.059),
        "BH-BH": (1.162, 1.974, 1.041, 59.88),
    }
    paper_3p5PN = {
        "NS-NS": (0.5193, 1.279, 0.0133, 1.319),
        "NS-BH": (0.9966, 0.9268, 0.0679, 1.457),
        "BH-BH": (2.078, 1.161, 0.5241, 5.739),
    }

    for name, (m1, m2) in systems.items():
        M = (m1 + m2) * MSUN_SEC
        eta = m1 * m2 / (m1 + m2) ** 2
        Mc = eta ** 0.6 * M

        df = TableIRecreate(
            t_c_val=0.0, phi_c_val=0.0,
            M_c_val=Mc, eta_val=eta,
            chi1_val=0.0, chi2_val=0.0,
            kappa_s_val=1.0, kappa_a_val=0.0,  # irrelevant since chi1=chi2=0
            should_print_result=False
        )
        print(f"\n=== {name} ===")
        print(df.round(4))
        #print("paper 1PN  :", paper_1PN[name])
        #print("paper 3.5PN:", paper_3p5PN[name])


if __name__ == "__main__":
    _validate_against_paper()