"""
run.py -- the only entry point.  Usage:  python run.py <command>

  psd-fits       aLIGO fits vs numerical PSD + SNR density           (old Plotting.py)
  fisher-table   Table I: errors vs PN order for the boson-star pair  (old FisherKappaPlay.py)
  fig2           recreate Fig. 2 of arXiv:1811.00317                  (old Fig2 script)
  mcmc           emcee run + corner plot, saves output/mcmc_samples.npy  (old BayesPlay.py)
  compare        Fisher ellipses over MCMC corner plot                (old CompareFisherBayes.py)
  psd-gw150914   analytical vs measured PSDs (needs PyCBC + internet) (old Analytical.py demo)
"""
import argparse

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

import bayes
import boson_star_data as bsd
import config as cfg
import fisher
import plots
import psd


def _fiducial():
    return bsd.fiducial_from_rows()


def _fisher_args(fid):
    return dict(Mc=fid["Mc"], eta=fid["eta"], kappa_s=fid["kappa_s"], kappa_a=fid["kappa_a"],
                chi1=cfg.CHI1, chi2=cfg.CHI2)


def _bayes_model(fid):
    return bayes.BayesModel(**_fisher_args(fid))


# ----------------------------------------------------------------------------
def cmd_psd_fits():
    df = psd.load_numerical_psd()
    density = psd.snr_density(df["freq"], df["psd"])
    print(f"Signal to noise ratio of aLIGO (A=1): {trapezoid(density, df['freq'])}")
    print("saved", plots.plot_aligo_fits(df))
    print("saved", plots.plot_snr_density(df["freq"], density))


def cmd_fisher_table():
    fid = _fiducial()
    print(f"body1: m={fid['m1']:.4f}, kappa={fid['kappa1']:.4f} | "
          f"body2: m={fid['m2']:.4f}, kappa={fid['kappa2']:.4f} | "
          f"kappa_s={fid['kappa_s']:.4f}, kappa_a={fid['kappa_a']:.4f}, eta={fid['eta']:.8f}")
    print("saved", plots.plot_kappa_mass(bsd.load_mass_kappa_table(),
                                         fid["m1"], fid["kappa1"], fid["m2"], fid["kappa2"]))
    df = fisher.error_table_by_pn_order(**_fisher_args(fid))
    out = cfg.OUTPUT_DIR / "TableRecreated.csv"
    df.to_csv(out, float_format="%.4g")
    print("saved", out)


def cmd_fig2():
    masses = cfg.FIG2_TOTAL_MASSES
    records, top, bottom = [], [], []
    for q, label, color, marker in cfg.FIG2_TOP_CASES:
        print("top panel:", label)
        y = fisher.sweep_delta_kappa_s(masses, q, cfg.CHI1, cfg.CHI2)
        top.append((label, y, color, marker))
        records += [dict(panel="top", q=q, chi1=cfg.CHI1, chi2=cfg.CHI2, M_total=M, dkappa_s=v)
                    for M, v in zip(masses, y)]
    for chi1, chi2, color, marker in cfg.FIG2_BOTTOM_CASES:
        label = rf"$\chi_1$={chi1}, $\chi_2$={chi2}"
        print("bottom panel:", label)
        y = fisher.sweep_delta_kappa_s(masses, cfg.FIG2_BOTTOM_Q, chi1, chi2)
        bottom.append((label, y, color, marker))
        records += [dict(panel="bottom", q=cfg.FIG2_BOTTOM_Q, chi1=chi1, chi2=chi2, M_total=M, dkappa_s=v)
                    for M, v in zip(masses, y)]
    print("saved", plots.plot_fig2(masses, top, bottom))
    csv = cfg.OUTPUT_DIR / "Fig2_data.csv"
    pd.DataFrame(records).to_csv(csv, index=False, float_format="%.5g")
    print("saved", csv)


def cmd_mcmc():
    fid = _fiducial()
    model = _bayes_model(fid)
    cov = fisher.fisher_covariance(**_fisher_args(fid))
    sampler = bayes.run_mcmc(model, fisher_cov=cov)
    flat = bayes.flatten_chain(sampler)
    np.save(cfg.MCMC_SAMPLES_FILE, flat)
    print("saved", cfg.MCMC_SAMPLES_FILE)
    print("saved", plots.plot_corner(flat, cfg.PARAM_LABELS, model.theta_true, "kappa_corner.png"))


def cmd_compare():
    fid = _fiducial()
    theta_true = _bayes_model(fid).theta_true
    flat = np.load(cfg.MCMC_SAMPLES_FILE)
    cov_full = fisher.fisher_covariance(**_fisher_args(fid))

    keep = [2, 3, 4]                                   # M_c, eta, kappa_s
    labels = [cfg.PARAM_LABELS[i] for i in keep]
    cov = cov_full[np.ix_(keep, keep)]
    flat, truth = flat[:, keep], theta_true[keep]

    print("\nInjected vs. recovered parameters (M_c, eta, kappa_s):\n")
    print(bayes.comparison_table(flat, truth, np.sqrt(np.diag(cov)), labels).round(6))
    print("saved", plots.plot_fisher_vs_bayes(flat, labels, truth, cov))


def cmd_psd_gw150914():
    measured = psd.measured_psds_gw150914()
    freqs_s, h_net = psd.synthetic_heavy_bh_strain_squared()
    print("saved", plots.plot_psd_vs_gw150914(measured, freqs_s, h_net))


COMMANDS = {
    "psd-fits": cmd_psd_fits, "fisher-table": cmd_fisher_table, "fig2": cmd_fig2,
    "mcmc": cmd_mcmc, "compare": cmd_compare, "psd-gw150914": cmd_psd_gw150914,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=COMMANDS)
    COMMANDS[ap.parse_args().command]()
