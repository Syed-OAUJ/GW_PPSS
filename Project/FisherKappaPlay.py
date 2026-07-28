from Analytical import hAISSKappa, AISSKappa
from Plotting import Sh_new_numeric, Sh_new_func, f as f_plot

import sympy as sp
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt

##############################################################################
# UNIT CONVENTION: all masses in seconds (G=c=1), not solar masses.
##############################################################################
MSUN_SEC = 4.925491025543576e-6  # G*M_sun/c^3, seconds


##############################################################################
# SYMBOLIC VARIABLES
##############################################################################
# kappa1_ is now a Fisher-matrix parameter (body 1's quadrupole-monopole
# parameter). kappa2_ (body 2's) stays a FIXED fiducial number -- it is
# substituted numerically like chi1_/chi2_, never differentiated.
f_, t_c_, phi_c_, M_c_, eta_, kappa1_, kappa2_, chi1_, chi2_ = sp.symbols(
    "f t_c phi_c M_c eta kappa1 kappa2 chi1 chi2"
)


##############################################################################
# KAPPA(CHI) DATA -- tamaraevst/Spin-induced-quadrupole-moments-of-boson-stars
##############################################################################

KAPPA_DATA_PATH = "data/solitonic_sigma_0.04.dat"


def load_kappa_chi_table(path=KAPPA_DATA_PATH):
    """
    Columns: freq, mass, ang, radius99, radius95, compactness, kappa.
    There's no explicit dimensionless spin column, so chi = ang/mass^2
    (J/M^2 in G=c=1 units) is computed here.
    """
    df = pd.read_csv(
        path, sep=r"\s+", comment="#",
        names=["freq", "mass", "ang", "radius99", "radius95", "compactness", "kappa"]
    )
    df["chi"] = df["ang"] / df["mass"] ** 2
    df = df.drop_duplicates(subset=["chi"]).sort_values("chi").reset_index(drop=True)
    return df


def kappa_from_chi(chi_query, table=None):
    """
    Nearest-neighbour lookup of kappa at a given chi.
    NOTE: this sequence is NOT single-valued in chi -- several boson-star
    branches (different compactness/mass) share the same chi -- so this
    picks whichever tabulated point is closest in chi, rather than
    interpolating across branches that would be ill-defined.
    """
    if table is None:
        table = load_kappa_chi_table()
    idx = (table["chi"] - chi_query).abs().idxmin()
    row = table.loc[idx]
    return float(row["kappa"]), float(row["chi"])


def plot_kappa_chi(fiducial_points=None, table=None,
                    outpath="kappa_vs_chi.png"):
    """fiducial_points: list of (chi, kappa, label, color, marker) to overlay."""
    if table is None:
        table = load_kappa_chi_table()
    plt.figure(figsize=(7.5, 5.5))
    plt.scatter(table["chi"], table["kappa"], s=6, alpha=0.45, color="tab:blue",
                label="solitonic boson star sequence (sigma=0.04)")
    for chi_p, kappa_p, label, color, marker in (fiducial_points or []):
        plt.scatter([chi_p], [kappa_p], color=color, marker=marker, s=60,
                    zorder=5, label=label)
    plt.xlabel(r"$\chi = J/M^2$")
    plt.ylabel(r"$\kappa$ (spin-induced quadrupole parameter)")
    plt.title("Boson star spin-induced quadrupole parameter vs. dimensionless spin")
    plt.legend(fontsize=9)
    plt.grid(True, ls=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    return outpath


##############################################################################
# AMPLITUDE: unit-amplitude h -> physical A giving SNR = rho
##############################################################################

def calc_amplitude(Sh_func, freqs, rho=10.0):
    integral = trapezoid(freqs ** (-7 / 3) / Sh_func(freqs), freqs)
    return rho / (2.0 * np.sqrt(integral))


def f_lso(M_val):
    """Schwarzschild LSO frequency, M_val in seconds."""
    return 1.0 / (6.0 ** 1.5 * np.pi * M_val)


def make_freq_grid(M_val, fs=20.0, f_upper_cap=None, n=500):
    f_max = f_lso(M_val)
    if f_upper_cap is not None:
        f_max = min(f_max, f_upper_cap)
    if f_max <= fs:
        raise ValueError(f"f_lso ({f_max:.1f} Hz) <= fs ({fs} Hz): system merges below band.")
    return np.logspace(np.log10(fs), np.log10(f_max), n)


##############################################################################
# helper: evaluate a single symbolic expression on the frequency grid,
# broadcasting literal constants (e.g. an exact symbolic 0, which happens
# for d(phase)/d(kappa1) below 2PN, since kappa first enters at alpha4_)
##############################################################################

def _eval_expr_on_freqs(expr, freqs):
    fn = sp.lambdify(f_, expr, modules="numpy")
    val = fn(freqs)
    val = np.asarray(val, dtype=np.complex128)
    if val.shape != freqs.shape:
        val = np.broadcast_to(val, freqs.shape).copy()
    return val


##############################################################################
# FISHER MATRIX  (theta = t_c, phi_c, M_c, eta, kappa1)
##############################################################################

def CreateFisherMatrice(
        t_c_val, phi_c_val, M_c_val, eta_val,
        kappa1_val, kappa2_val,
        chi1_val, chi2_val,
        max_order=7,
        fs=20.0, f_upper_cap=None, n_freqs=500,
        should_print_result=False
):
    theta = [t_c_, phi_c_, M_c_, eta_, kappa1_]

    M_total_ = M_c_ / eta_ ** sp.Rational(3, 5)

    kappa_s_expr = (kappa1_ + kappa2_) / 2
    kappa_a_expr = (kappa1_ - kappa2_) / 2

    psi = AISSKappa(
        f_, M_total_, eta_,
        chi1_, chi2_,
        kappa_s_expr, kappa_a_expr,
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
        kappa1_: kappa1_val, kappa2_: kappa2_val,
    }
    fisher = fisher.subs(fiducial)

    M_val = M_c_val / eta_val ** 0.6
    freqs = make_freq_grid(M_val, fs=fs, f_upper_cap=f_upper_cap, n=n_freqs)

    # --- element-wise lambdify (fixes the ragged-array crash) ---
    n = len(theta)
    fisher_vals = np.empty((n, n, len(freqs)), dtype=np.complex128)
    for i in range(n):
        for j in range(i, n):
            fisher_vals[i, j, :] = _eval_expr_on_freqs(fisher[i, j], freqs)
            fisher_vals[j, i, :] = fisher_vals[i, j, :]

    fisher_total_unit = np.asarray(trapezoid(fisher_vals.real, freqs, axis=2), dtype=float)

    amplitude = calc_amplitude(Sh_new_func, freqs)
    fisher_total = amplitude ** 2 * fisher_total_unit

    # --- handle kappa1 carrying zero information below 2PN ---
    kappa_idx = theta.index(kappa1_)
    kappa_uninformative = np.allclose(fisher_total[kappa_idx, :], 0.0)

    if kappa_uninformative:
        keep = [k for k in range(n) if k != kappa_idx]
        cov_sub = np.linalg.inv(fisher_total[np.ix_(keep, keep)])
        covariance = np.full((n, n), np.nan)
        for a, ia in enumerate(keep):
            for b, ib in enumerate(keep):
                covariance[ia, ib] = cov_sub[a, b]
    else:
        covariance = np.linalg.inv(fisher_total)

    if should_print_result:
        print("\nFisher matrix\n", pd.DataFrame(fisher_total).round(5))
        print("\nCovariance\n", pd.DataFrame(covariance).round(5))
        if not kappa_uninformative:
            print("\nCheck Gamma @ Sigma = I\n", np.round(fisher_total @ covariance, 5))
        else:
            print("\nNote: kappa1 has zero Fisher information at this PN order "
                  "(first enters at 2PN via alpha4) -> Delta kappa1 = NaN")

    return covariance


##############################################################################
# TABLE  (now with an extra Delta kappa1 column)
##############################################################################

_PN_LABELS = {2: "1PN", 3: "1.5PN", 4: "2PN", 5: "2.5PN", 6: "3PN", 7: "3.5PN"}


def TableIRecreate(
        t_c_val, phi_c_val, M_c_val, eta_val,
        chi1_val, chi2_val,
        kappa1_val, kappa2_val,
        fs=20.0, f_upper_cap=None,
        should_print_result=True
):
    rows, labels = [], []
    for order in range(2, 8):
        covariance = CreateFisherMatrice(
            t_c_val=t_c_val, phi_c_val=phi_c_val,
            M_c_val=M_c_val, eta_val=eta_val,
            kappa1_val=kappa1_val, kappa2_val=kappa2_val,
            chi1_val=chi1_val, chi2_val=chi2_val,
            max_order=order, fs=fs, f_upper_cap=f_upper_cap,
        )
        sigma = np.sqrt(np.diag(covariance))
        rows.append([
            sigma[0] * 1000,
            sigma[1],
            100 * sigma[2] / M_c_val,
            100 * sigma[3] / eta_val,
            sigma[4],                     # Delta kappa1, absolute
        ])
        labels.append(_PN_LABELS[order])

    df = pd.DataFrame(
        rows, index=labels,
        columns=[r"$\Delta t_c$ (ms)", r"$\Delta\phi_c$",
                 r"$\Delta M_c/M_c$ (%)", r"$\Delta\eta/\eta$ (%)",
                 r"$\Delta\kappa_1$"]
    )
    if should_print_result:
        print(df.round(4))
    return df


def load_mass_kappa_table(path=KAPPA_DATA_PATH):
    return pd.read_csv(
        path, sep=r"\s+", comment="#",
        names=["freq", "mass", "ang", "radius99", "radius95", "compactness", "kappa"],
        usecols=["mass", "kappa"],
    )


def mass_kappa_from_rows(row1_idx=0, row2_idx=1, path=KAPPA_DATA_PATH):
    """
    kappa1/mass1 <- row1_idx (default: first row)
    kappa2/mass2 <- row2_idx (default: second row)
    """
    table = load_mass_kappa_table(path)
    r1, r2 = table.iloc[row1_idx], table.iloc[row2_idx]
    return {
        "mass1": float(r1["mass"]), "kappa1": float(r1["kappa"]),
        "mass2": float(r2["mass"]), "kappa2": float(r2["kappa"]),
    }


def plot_kappa_mass(m1, kappa1, m2, kappa2, table=None,
                     outpath="kappa_vs_mass.png"):
    if table is None:
        table = load_mass_kappa_table()
    plt.figure(figsize=(7.5, 5.5))
    plt.scatter(table["mass"], table["kappa"], s=6, alpha=0.35, color="tab:blue",
                label="full sequence (mass, kappa)")
    plt.scatter([m1], [kappa1], color="red", s=70, zorder=5,
                label=f"row 1 (body 1): m1={m1:.3f}, kappa1={kappa1:.3f}")
    plt.scatter([m2], [kappa2], color="darkorange", marker="D", s=70, zorder=5,
                label=f"row 2 (body 2): m2={m2:.3f}, kappa2={kappa2:.3f}")
    plt.xlabel("mass")
    plt.ylabel("kappa")
    plt.title("Boson star kappa vs mass (solitonic, sigma=0.04)")
    plt.legend(fontsize=9)
    plt.grid(True, ls=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    return outpath


##############################################################################
# RUN
##############################################################################

if __name__ == "__main__":
    vals = mass_kappa_from_rows()  # row 0 -> body 1, row 1 -> body 2
    m1, kappa1_val = vals["mass1"], vals["kappa1"]
    m2, kappa2_val = vals["mass2"], vals["kappa2"]
    chi1_val, chi2_val = 0.9, 0.8  # fixed constants, unrelated to the data rows
    #m1, m2 = 10, 10
    M = (m1 + m2) * MSUN_SEC
    eta = m1 * m2 / (m1 + m2) ** 2
    #eta = 0.25
    Mc = eta ** 0.6 * M

    print(f"row1 -> m1={m1:.4f}, kappa1={kappa1_val:.4f}")
    print(f"row2 -> m2={m2:.4f}, kappa2={kappa2_val:.4f}")
    print(f"eta = {eta:.8f}   (1 - 4*eta = {1 - 4*eta:.6e}) "
          f"-- close to 0.25 since m1~m2 here, expect large-but-finite eta-sensitivity")

    plot_kappa_mass(m1, kappa1_val, m2, kappa2_val)
    print("saved kappa_vs_mass.png")

    df = TableIRecreate(
        t_c_val=0.0, phi_c_val=0.0,
        M_c_val=Mc, eta_val=eta,
        chi1_val=chi1_val, chi2_val=chi2_val,
        kappa1_val=kappa1_val, kappa2_val=kappa2_val,
    )
    """
    df = TableIRecreate(
        t_c_val=0.0, phi_c_val=0.0,
        M_c_val=Mc, eta_val=eta,
        chi1_val=0, chi2_val=0,
        kappa1_val=1, kappa2_val=1,
    )
    """
    df.to_csv("TableRecreated.csv", index=False, float_format="%.4g")
