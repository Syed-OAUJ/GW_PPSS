"""plots.py -- every figure the project makes. Functions take data, save a PNG, return its path."""
import matplotlib.pyplot as plt
import numpy as np

import config as cfg
import psd


def _save(fig, name, **kw):
    path = cfg.OUTPUT_DIR / name
    fig.savefig(path, dpi=150, **kw)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------
# PSDs and SNR
# ----------------------------------------------------------------------------
def plot_aligo_fits(df_numeric, name="aligo_psd_comparison.png"):
    f_new = np.logspace(np.log10(cfg.ALIGO_NEW_FIT["fs"]), np.log10(2000), 500)
    f_old = np.logspace(np.log10(cfg.ALIGO_OLD_FIT["fs"]), np.log10(2000), 500)
    fig = plt.figure(figsize=(7, 6))
    plt.loglog(f_new, psd.sh_aligo_new(f_new), "k-", lw=2, label=r"Adv. LIGO (Eq. 3.7, $f_0=215$ Hz)")
    plt.loglog(f_old, psd.sh_aligo_old(f_old), "b--", lw=2, label=r"Adv. LIGO (older fit, $f_0=70$ Hz)")
    plt.loglog(df_numeric["freq"], df_numeric["psd"], label="Data")
    plt.xlabel("Frequency (Hz)"); plt.ylabel(r"$S_h(f)$  (Hz$^{-1}$)")
    plt.title("Comparison of Advanced LIGO noise PSD fits")
    plt.legend(); plt.grid(True, which="both", ls=":", alpha=0.6); plt.xlim(*cfg.F_PLOT_RANGE)
    plt.tight_layout()
    return _save(fig, name)


def plot_snr_density(freq, density, name="SNR_density.png"):
    fig = plt.figure(figsize=(7, 6))
    plt.loglog(freq, density, label="SNR Aligo density")
    plt.xlabel("Frequency (Hz)"); plt.ylabel("SNR density")
    plt.title("SNR Density Comparison")
    plt.legend(); plt.grid(True, which="both", ls=":", alpha=0.6); plt.xlim(*cfg.F_PLOT_RANGE)
    plt.tight_layout()
    return _save(fig, name)


def plot_psd_vs_gw150914(measured, freqs_synth, h_net_synth, name="PSD_plot.png"):
    freqs = np.logspace(0, 4, 2000)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(freqs, psd.sh_einstein_telescope(freqs), label="PSDET (analytical)")
    ax.loglog(freqs, psd.sh_ajith(freqs), label="PSDAjith (analytical)")
    ax.loglog(freqs_synth, h_net_synth, label="Synthetic Heavy BH")
    for ifo, p in measured.items():
        fr = p.sample_frequencies.numpy()
        m = (fr >= 1) & (fr <= 1e4)
        ax.loglog(fr[m], p.numpy()[m], label=f"{ifo} measured PSD (GW150914)", alpha=0.8)
    ax.set_xlabel("Frequency f [Hz]"); ax.set_ylabel(r"PSD$(f)$  [strain$^2$/Hz]")
    ax.set_title("Analytical vs. measured detector PSDs (GW150914)")
    ax.grid(True, which="both", ls="--", alpha=0.5); ax.legend()
    fig.tight_layout()
    return _save(fig, name)


# ----------------------------------------------------------------------------
# Boson-star tables
# ----------------------------------------------------------------------------
def plot_kappa_mass(table, m1, k1, m2, k2, name="kappa_vs_mass.png"):
    fig = plt.figure(figsize=(7.5, 5.5))
    plt.scatter(table["mass"], table["kappa"], s=6, alpha=0.35, color="tab:blue",
                label="full sequence (mass, kappa)")
    plt.scatter([m1], [k1], color="red", s=70, zorder=5, label=f"body 1: m1={m1:.3f}, kappa1={k1:.3f}")
    plt.scatter([m2], [k2], color="darkorange", marker="D", s=70, zorder=5,
                label=f"body 2: m2={m2:.3f}, kappa2={k2:.3f}")
    plt.xlabel("mass"); plt.ylabel("kappa"); plt.title("Boson star kappa vs mass (solitonic, sigma=0.04)")
    plt.legend(fontsize=9); plt.grid(True, ls=":", alpha=0.6); plt.tight_layout()
    return _save(fig, name)


def plot_kappa_chi(table, fiducial_points=None, name="kappa_vs_chi.png"):
    """fiducial_points: list of (chi, kappa, label, color, marker)."""
    fig = plt.figure(figsize=(7.5, 5.5))
    plt.scatter(table["chi"], table["kappa"], s=6, alpha=0.45, color="tab:blue",
                label="solitonic boson star sequence (sigma=0.04)")
    for chi, kappa, label, color, marker in (fiducial_points or []):
        plt.scatter([chi], [kappa], color=color, marker=marker, s=60, zorder=5, label=label)
    plt.xlabel(r"$\chi = J/M^2$"); plt.ylabel(r"$\kappa$ (spin-induced quadrupole parameter)")
    plt.title("Boson star spin-induced quadrupole parameter vs. dimensionless spin")
    plt.legend(fontsize=9); plt.grid(True, ls=":", alpha=0.6); plt.tight_layout()
    return _save(fig, name)


# ----------------------------------------------------------------------------
# Fisher forecasts
# ----------------------------------------------------------------------------
def plot_fig2(masses, top_curves, bottom_curves, name="Fig2_recreated.png"):
    """top_curves: [(label, y, color, marker)], bottom_curves likewise."""
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(6.5, 10), sharex=True)
    for ax, curves in ((ax_top, top_curves), (ax_bot, bottom_curves)):
        for label, y, color, marker in curves:
            ax.plot(masses, y, linestyle="none", marker=marker, mfc="none",
                    mec=color, mew=1.5, ms=8, label=label)
        ax.set_yscale("log"); ax.set_ylabel(r"$\Delta\kappa_s$"); ax.set_xlim(8, 82)
        ax.grid(True, which="both", ls=":", alpha=0.5); ax.legend(fontsize=9, loc="upper right")
    ax_bot.set_xlabel(r"Total Mass ($M_\odot$)")
    fig.tight_layout()
    return _save(fig, name)


# ----------------------------------------------------------------------------
# Posteriors
# ----------------------------------------------------------------------------
def plot_corner(flat_samples, labels, truths, name="kappa_corner.png"):
    import corner
    fig = corner.corner(flat_samples, labels=labels, truths=list(truths),
                        quantiles=[0.16, 0.5, 0.84], show_titles=True)
    return _save(fig, name)


def _cov_ellipse(ax, mean_xy, cov2x2, n_std=1.0, **kw):
    from matplotlib.patches import Ellipse
    vals, vecs = np.linalg.eigh(cov2x2)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2 * n_std * np.sqrt(np.clip(vals, 0, None))
    ax.add_patch(Ellipse(xy=mean_xy, width=w, height=h, angle=angle, **kw))


def plot_fisher_vs_bayes(flat_samples, labels, truths, fisher_cov, name="fisher_vs_bayes_corner.png"):
    """MCMC corner plot (black) with Fisher Gaussian overlaid (red); values go in the legend."""
    import corner
    from matplotlib.lines import Line2D

    ndim = len(labels)
    truths = np.asarray(truths)
    sig = np.sqrt(np.diag(fisher_cov))
    med = np.median(flat_samples, axis=0)
    lo, hi = np.percentile(flat_samples, [16, 84], axis=0)

    fig = corner.corner(flat_samples, labels=labels, truths=list(truths), truth_color="blue",
                        quantiles=[0.16, 0.5, 0.84], show_titles=True,
                        hist_kwargs={"density": True}, color="black")
    axes = np.array(fig.axes).reshape((ndim, ndim))

    for i in range(ndim):
        xs = np.linspace(*axes[i, i].get_xlim(), 300)
        pdf = np.exp(-0.5 * ((xs - truths[i]) / sig[i]) ** 2) / (sig[i] * np.sqrt(2 * np.pi))
        axes[i, i].plot(xs, pdf, color="red", lw=1.5)
        for j in range(i):
            _cov_ellipse(axes[i, j], (truths[j], truths[i]), fisher_cov[np.ix_([j, i], [j, i])],
                         edgecolor="red", facecolor="none", lw=1.5, zorder=10)

    fmt = lambda x: f"{x:.4g}"
    bare = [l.strip("$") for l in labels]
    inj = "\n".join(f"${b} = {fmt(v)}$" for b, v in zip(bare, truths))
    rec = "\n".join(f"${b} = {fmt(m)}^{{+{fmt(h - m)}}}_{{-{fmt(m - l)}}}$"
                    for b, m, l, h in zip(bare, med, lo, hi))
    handles = [Line2D([0], [0], color="red", lw=1.5, label="Fisher (Gaussian, 1-sigma)"),
               Line2D([0], [0], color="none", label=f"Injected:\n{inj}"),
               Line2D([0], [0], color="none", label=f"Recovered (median, 1-sigma):\n{rec}")]
    ax = axes[0, ndim - 1]
    ax.set_visible(True); ax.axis("off")
    ax.legend(handles=handles, fontsize=8, loc="center", frameon=True)
    return _save(fig, name, bbox_inches="tight")
