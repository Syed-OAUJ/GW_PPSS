"""
Recreate Fig. 2 of Krishnendu, Mishra, Arun (arXiv:1811.00317):
1-sigma error on kappa_s vs total mass.

  top panel   : q = 1.2, 3, 5 with (chi1, chi2) = (0.9, 0.8)
  bottom panel: q = 1.2 with four spin configurations
                (0.9, 0.8), (0.9, -0.8), (-0.9, 0.8), (-0.9, -0.8)

Put this file next to FisherKappaPlay.py / Analytical.py / Plotting.py.
Fiducial: binary black hole, kappa_s = 1, kappa_a = 0.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from FisherKappaPlay import CreateFisherMatrice, MSUN_SEC

# ----------------------------------------------------------------------------
# settings
# ----------------------------------------------------------------------------
TOTAL_MASSES = np.arange(10, 81, 10)       # Msun, as in the paper's x-axis
KAPPA_S_BH, KAPPA_A_BH = 1.0, 0.0
MAX_ORDER = 7                              # 3.5PN phasing
FS = 20.0                                  # low-frequency cutoff [Hz] (your aLIGO PSD)

_cache = {}


def delta_kappa_s(M_tot_msun, q, chi1, chi2):
    """1-sigma error on kappa_s for given total mass [Msun], q = m1/m2 >= 1, spins."""
    key = (float(M_tot_msun), float(q), float(chi1), float(chi2))
    if key in _cache:
        return _cache[key]

    eta = q / (1.0 + q) ** 2
    M = M_tot_msun * MSUN_SEC              # seconds (G=c=1)
    Mc = eta ** 0.6 * M

    try:
        cov = CreateFisherMatrice(
            t_c_val=0.0, phi_c_val=0.0, M_c_val=Mc, eta_val=eta,
            kappa_s_val=KAPPA_S_BH, kappa_a_val=KAPPA_A_BH,
            chi1_val=chi1, chi2_val=chi2,
            max_order=MAX_ORDER, fs=FS,
        )
        val = float(np.sqrt(cov[4, 4]))    # theta = [t_c, phi_c, M_c, eta, kappa_s]
    except (ValueError, np.linalg.LinAlgError) as err:
        print(f"  skipped M={M_tot_msun}, q={q}, chi=({chi1},{chi2}): {err}")
        val = np.nan

    _cache[key] = val
    return val


def sweep(q, chi1, chi2):
    return np.array([delta_kappa_s(M, q, chi1, chi2) for M in TOTAL_MASSES])


# ----------------------------------------------------------------------------
# curves
# ----------------------------------------------------------------------------
top_cases = [  # (q, label, color, marker)
    (1.2, "q=1.2", "tab:orange", "D"),
    (3.0, "q=3",   "gray",       "s"),
    (5.0, "q=5",   "black",      "v"),
]
bottom_cases = [  # (chi1, chi2, color, marker)
    (0.9,  0.8, "tab:orange", "D"),
    (0.9, -0.8, "gray",       "s"),
    (-0.9, 0.8, "black",      "v"),
    (-0.9, -0.8, "olive",     "^"),
]

records = []

fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(6.5, 10), sharex=True)

# top panel
for q, label, color, marker in top_cases:
    print(f"top panel: {label}")
    y = sweep(q, 0.9, 0.8)
    ax_top.plot(TOTAL_MASSES, y, linestyle="none", marker=marker, mfc="none",
                mec=color, mew=1.5, ms=8, label=label)
    for M, v in zip(TOTAL_MASSES, y):
        records.append(dict(panel="top", q=q, chi1=0.9, chi2=0.8, M_total=M, dkappa_s=v))

# bottom panel (q = 1.2)
for chi1, chi2, color, marker in bottom_cases:
    label = rf"$\chi_1$={chi1}, $\chi_2$={chi2}"
    print(f"bottom panel: {label}")
    y = sweep(1.2, chi1, chi2)
    ax_bot.plot(TOTAL_MASSES, y, linestyle="none", marker=marker, mfc="none",
                mec=color, mew=1.5, ms=8, label=label)
    for M, v in zip(TOTAL_MASSES, y):
        records.append(dict(panel="bottom", q=1.2, chi1=chi1, chi2=chi2, M_total=M, dkappa_s=v))

for ax in (ax_top, ax_bot):
    ax.set_yscale("log")
    ax.set_ylabel(r"$\Delta\kappa_s$")
    ax.set_xlim(8, 82)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=9, loc="upper right")
ax_bot.set_xlabel(r"Total Mass ($M_\odot$)")

fig.tight_layout()
fig.savefig("Fig2_recreated.png", dpi=150)
pd.DataFrame(records).to_csv("Fig2_data.csv", index=False, float_format="%.5g")
print("saved Fig2_recreated.png and Fig2_data.csv")