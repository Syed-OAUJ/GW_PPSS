import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import corner

from BayesPlay import (
    theta_true, PARAM_LABELS, Mc_val, eta_val, kappa_s_val, kappa_a_val,
    chi1_val, chi2_val, t_c_val, phi_c_val,
)
from FisherKappaPlay import CreateFisherMatrice

##############################################################################
# Load the MCMC samples produced by BayesKappaPlay.py, and the Fisher
# covariance at the same fiducial point (cheap to recompute -- no MCMC needed)
##############################################################################

flat_samples = np.load("mcmc_samples.npy")

fisher_cov = CreateFisherMatrice(
    t_c_val=t_c_val, phi_c_val=phi_c_val,
    M_c_val=Mc_val, eta_val=eta_val,
    kappa_s_val=kappa_s_val, kappa_a_val=kappa_a_val,
    chi1_val=chi1_val, chi2_val=chi2_val,
    max_order=7,
)

ndim = len(theta_true)


def cov_ellipse(ax, mean_xy, cov2x2, n_std=1.0, **kwargs):
    """1-sigma (by default) confidence ellipse for a 2D Gaussian with the
    given 2x2 covariance block, centered at mean_xy=(x, y)."""
    vals, vecs = np.linalg.eigh(cov2x2)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(np.clip(vals, 0, None))
    ell = Ellipse(xy=mean_xy, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ell)
    return ell


##############################################################################
# Corner plot of the Bayes posterior, with the Fisher-predicted Gaussian
# overlaid: a 1D Gaussian curve on each diagonal panel, a 1-sigma ellipse
# on each lower off-diagonal panel.
##############################################################################

fig = corner.corner(
    flat_samples, labels=PARAM_LABELS, truths=list(theta_true),
    quantiles=[0.16, 0.5, 0.84], show_titles=True,
    hist_kwargs={"density": True},   # needed so the Gaussian overlay is on the same y-scale
    color="tab:blue",
)

axes = np.array(fig.axes).reshape((ndim, ndim))

for i in range(ndim):
    ax = axes[i, i]
    sigma_i = np.sqrt(fisher_cov[i, i])
    xlim = ax.get_xlim()
    xs = np.linspace(*xlim, 300)
    pdf = np.exp(-0.5 * ((xs - theta_true[i]) / sigma_i) ** 2) / (sigma_i * np.sqrt(2 * np.pi))
    ax.plot(xs, pdf, color="red", lw=1.5, label="Fisher (Gaussian)" if i == 0 else None)

    for j in range(i):
        ax2 = axes[i, j]
        cov2x2 = fisher_cov[np.ix_([j, i], [j, i])]
        cov_ellipse(ax2, (theta_true[j], theta_true[i]), cov2x2, n_std=1.0,
                    edgecolor="red", facecolor="none", lw=1.5, zorder=10)

axes[0, 0].legend(fontsize=8, loc="upper right")

fig.savefig("fisher_vs_bayes_corner.png", dpi=150)
plt.show()