from Analytical import hAISSKappa, AISSKappa
from Plotting import Sh_new_numeric, Sh_new_func, freqs_new, f as f_plot

import sympy as sp
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid


##############################################################################
# SYMBOLIC VARIABLES
##############################################################################

f_, t_c_, phi_c_, M_c_, eta_, kappa_a_, kappa_s_, chi1_, chi2_ = sp.symbols(
    "f t_c phi_c M_c eta kappa_a kappa_s chi1 chi2"
)


##############################################################################
# AMPLITUDE
##############################################################################

def calc_amplitude(Sh_new_func, freqs, rho=10):
    integral = trapezoid(freqs**(-7/3) / Sh_new_func(freqs), freqs)
    return rho / np.sqrt(integral)


##############################################################################
# FISHER MATRIX
##############################################################################

def CreateFisherMatrice(
        f_, t_c_, phi_c_, M_c_, eta_, kappa_a_, kappa_s_, chi1_, chi2_,
        t_c_val, phi_c_val, M_c_val, eta_val,
        kappa_a_val, kappa_s_val,
        chi1_val, chi2_val,
        max_order=7,
        should_print_result=False
):

    theta = [t_c_, phi_c_, M_c_, eta_]

    ##########################################################################
    # symbolic waveform
    ##########################################################################

    h = hAISSKappa(
        f_, M_c_, eta_,
        t_c_, phi_c_,
        chi1_, chi2_,
        kappa_s_, kappa_a_,
        max_order
    )

    phase = AISSKappa(
        f_, M_c_, eta_,
        chi1_, chi2_,
        kappa_s_, kappa_a_,
        max_order
    )

    grad = [
        sp.I * h * sp.diff(phase, par)
        for par in theta
    ]

    ##########################################################################
    # PSD
    ##########################################################################

    Sh = Sh_new_numeric.subs(f_plot, f_)

    ##########################################################################
    # symbolic Fisher matrix
    ##########################################################################

    fisher = sp.zeros(len(theta))

    for i in range(len(theta)):
        for j in range(len(theta)):

            fisher[i, j] = (
                2 *
                (
                    sp.conjugate(grad[i]) * grad[j]
                    + grad[i] * sp.conjugate(grad[j])
                )
                / Sh
            )

    ##########################################################################
    # numerical substitutions
    ##########################################################################

    fiducial = {
        t_c_: t_c_val,
        phi_c_: phi_c_val,
        M_c_: M_c_val,
        eta_: eta_val,
        chi1_: chi1_val,
        chi2_: chi2_val,
        kappa_s_: kappa_s_val,
        kappa_a_: kappa_a_val
    }

    fisher = fisher.subs(fiducial)

    print("Remaining symbols:", fisher.free_symbols)

    ##########################################################################
    # lambdify
    ##########################################################################

    fisher_fun = sp.lambdify(f_, fisher, modules="numpy")

    raw = fisher_fun(freqs_new)

    fisher_vals = np.empty(
        (4, 4, len(freqs_new)),
        dtype=np.complex128
    )

    for i in range(4):
        for j in range(4):
            element = raw[i][j]

            if np.isscalar(element):
                fisher_vals[i, j, :] = element
            else:
                fisher_vals[i, j, :] = element

    ##########################################################################
    # integrate
    ##########################################################################

    fisher_total = trapezoid(
        fisher_vals.real,
        freqs_new,
        axis=2
    )

    fisher_total = np.asarray(fisher_total, dtype=float)

    covariance = np.linalg.inv(fisher_total)

    if should_print_result:

        print("\nFisher matrix\n")
        print(pd.DataFrame(fisher_total).round(5))

        print("\nCovariance\n")
        print(pd.DataFrame(covariance).round(5))

        print("\nCheck ΓΓ⁻¹\n")
        print(fisher_total @ covariance)

    return covariance


##############################################################################
# TABLE
##############################################################################

def TableIRecreate(
        t_c_val,
        phi_c_val,
        M_c_val,
        eta_val,
        chi1_val,
        chi2_val,
        kappa_s_val,
        kappa_a_val,
        max_alpha_index=7
):

    amplitude = calc_amplitude(
        Sh_new_func,
        freqs_new
    )

    rows = []

    for order in range(max_alpha_index):

        covariance = CreateFisherMatrice(
            f_, t_c_, phi_c_, M_c_, eta_,
            kappa_a_, kappa_s_,
            chi1_, chi2_,
            t_c_val=t_c_val,
            phi_c_val=phi_c_val,
            M_c_val=M_c_val,
            eta_val=eta_val,
            chi1_val=chi1_val,
            chi2_val=chi2_val,
            kappa_s_val=kappa_s_val,
            kappa_a_val=kappa_a_val,
            max_order=order
        )

        sigma = np.sqrt(np.diag(covariance) / amplitude)

        rows.append([
            sigma[0],
            sigma[1],
            sigma[2] / M_c_val,
            sigma[3] / eta_val
        ])

    df = pd.DataFrame(
        rows,
        columns=[
            r"$\Delta t_c$",
            r"$\Delta\phi_c$",
            r"$\Delta M_c/M_c$",
            r"$\Delta\eta/\eta$"
        ]
    )

    print(df.round(5))

    return df


##############################################################################
# NUMERICAL VALUES
##############################################################################
"""
Mc_val = 20.0
eta_val = 0.24
tc_val = 0.0
phic_val = 0.0
chi1_val = 0.0
chi2_val = 0.0
kappa_s_val = 1.0
kappa_a_val = 0.0

TableIRecreate(t_c_val=tc_val, phi_c_val=phic_val, M_c_val=Mc_val, eta_val=eta_val, chi1_val=chi1_val,
               chi2_val=chi2_val, kappa_s_val=kappa_s_val, kappa_a_val=kappa_a_val)
"""
print(f"amplitude:{calc_amplitude(Sh_new_func, freqs_new)}")
# Fisher value for (dh/dt_c)



h = hAISSKappa(f_, M_c_, eta_, t_c_, phi_c_, chi1_, chi2_, kappa_s_, kappa_a_, max_order=1)
psi_pn = AISSKappa(f_, M_c_, eta_, chi1_, chi2_, kappa_s_, kappa_a_, max_order=1)
full_phase = psi_pn + 2 * sp.pi * f_ * t_c_ - phi_c_

# sanity check: this should print 2*pi*f_
print("d(phase)/d(t_c):", sp.simplify(sp.diff(full_phase, t_c_)))

h_t = sp.I * h * sp.diff(full_phase, t_c_)
h_phi = sp.I * h * sp.diff(full_phase, phi_c_)

Sh = Sh_new_numeric.subs(f_plot, f_)
fisher = 2 * (sp.conjugate(h_t) * h_t + h_t * sp.conjugate(h_t)) / Sh
fisher_2 = 2 * (sp.conjugate(h_t) * h_phi + h_t * sp.conjugate(h_phi)) / Sh

fiducial = {
    M_c_: 20.0, eta_: 0.24, t_c_: 0.0, phi_c_: 0.0,
    chi1_: 0.0, chi2_: 0.0, kappa_s_: 1.0, kappa_a_: 0.0,
}
fisher_num = fisher.subs(fiducial)
fisher_2_num = fisher_2.subs(fiducial)
print("remaining free symbols for h_t:", fisher_num.free_symbols)  # should be {f_}
print("remaining free symbols for h_phi:", fisher_2_num.free_symbols)  # should be {f_}
fisher_fun = sp.lambdify(f_, fisher_num, modules="numpy")
fisher_val_check = trapezoid(fisher_fun(freqs_new).real, freqs_new)
print("Fisher[t_c, t_c] =", fisher_val_check)

fisher_2_fun = sp.lambdify(f_, fisher_2_num, modules="numpy")
fisher_2_val_check = trapezoid(fisher_2_fun(freqs_new).real, freqs_new)
print("Fisher_2[t_c, phi_c] =", fisher_2_val_check)