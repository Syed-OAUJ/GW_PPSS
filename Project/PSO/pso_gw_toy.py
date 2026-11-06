# %% [markdown]
# # PSO on a gravitational-wave test case (toy version of the real search)
# Same logic as `global_optimizer_inspiral.py`, with everything stripped to numpy:
# template -> matched filter -> cost = -max_t SNR -> swarm moves -> best template.
# Open in Jupyter (jupytext), VS Code, or run as a script. PSD below is an ANALYTIC PLACEHOLDER:
# swap `psd()` for your aLIGO / ET / CE text file (log-log interpolation).

# %%
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
FIG = "figs"; os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 10})
MSUN_S = 4.925491e-6            # G*Msun/c^3 in seconds (geometrised units: crucial, see guide)

# %% [markdown]
# ## 1. Frequency grid, PSD, template, matched filter
# %%
fs, T = 1024.0, 32.0
N = int(fs * T); df = 1.0 / T
f = np.fft.rfftfreq(N, 1.0 / fs)
flow, fmax = 20.0, 512.0
band = (f >= flow) & (f <= fmax)

def psd(f):
    """PLACEHOLDER aLIGO-like analytic PSD (1/Hz). Replace by your real PSD file."""
    x = np.maximum(f, 1.0) / 215.0
    S = 1e-48 * (x**-4.14 - 5*x**-2 + 111*(1 - x**2 + x**4/2) / (1 + x**2/2))
    return np.where(band, S, np.inf)
Sn = psd(f)

def template(Mc, eta, tc=0.0, phic=0.0):
    """Restricted TaylorF2 (amplitude f^-7/6, phase to 1PN). Mc in Msun."""
    Ms = Mc * eta**(-0.6) * MSUN_S                    # total mass in seconds
    fisco = 1.0 / (6**1.5 * np.pi * Ms)
    use = band & (f <= fisco)
    v = (np.pi * Ms * np.maximum(f, 1e-3))**(1/3)
    psi = (2*np.pi*f*tc - phic - np.pi/4
           + 3/(128*eta*v**5) * (1 + (20/9)*(743/336 + 11*eta/4)*v**2))
    h = np.where(use, np.maximum(f,1e-3)**(-7/6) * np.exp(1j*psi), 0.0)
    return h

def inner(a, b):
    m = band
    return 4*df*np.real(np.sum(a[m]*np.conj(b[m])/Sn[m]))

def snr_series(d, h):
    """|rho(t)| for all time shifts at once (one FFT) = what pycbc matched_filter does."""
    m = band & (h != 0)
    z = np.zeros(len(f), complex)
    z[m] = d[m]*np.conj(h[m])/Sn[m]
    full = np.zeros(N, complex); full[:len(f)] = z
    z_t = 4*df*N*np.fft.ifft(full)                    # complex SNR numerator vs time
    sigma = np.sqrt(inner(h, h))
    return np.abs(z_t)/sigma

def make_data(theta, snr_target, seed, tc=20.0, phic=0.7, signal=True):
    rng = np.random.default_rng(seed)
    n = np.sqrt(np.where(band, Sn, 0)*T/4)*(rng.normal(size=len(f)) + 1j*rng.normal(size=len(f)))
    n[~band] = 0
    if not signal: return n
    h = template(*theta, tc=tc, phic=phic)
    A = snr_target/np.sqrt(inner(h, h))
    return A*h + n

# truth: m1=12, m2=9 Msun
m1, m2 = 12.0, 9.0
Mc_t = (m1*m2)**0.6/(m1+m2)**0.2; eta_t = m1*m2/(m1+m2)**2
theta_t = (Mc_t, eta_t)
print("truth Mc=%.3f eta=%.4f" % theta_t)
d = make_data(theta_t, 20.0, seed=1)
print("rho at truth (max_t) = %.2f   (target 20)" % snr_series(d, template(*theta_t)).max())

def cost(x, data):
    return -snr_series(data, template(x[0], x[1])).max()

lb = np.array([6.0, 0.10]); ub = np.array([14.0, 0.25])

# %% [markdown]
# ## 2. The cost landscape (what the swarm is exploring)
# %%
Mcs = np.linspace(lb[0], ub[0], 90); etas = np.linspace(lb[1], ub[1], 70)
L = np.array([[-cost((a, b), d) for a in Mcs] for b in etas])
i = np.unravel_index(L.argmax(), L.shape)
print("grid max rho=%.2f at Mc=%.2f eta=%.3f  (%d evaluations)" % (L.max(), Mcs[i[1]], etas[i[0]], L.size))
fig, ax = plt.subplots(figsize=(5.6, 4))
c = ax.contourf(Mcs, etas, L, 30, cmap="viridis"); plt.colorbar(c, label=r"$\max_t\,\rho(t)$")
ax.plot(*theta_t, "r*", ms=13, label="truth"); ax.set_xlabel(r"$\mathcal{M}_c\ [M_\odot]$"); ax.set_ylabel(r"$\eta$"); ax.legend()
ax.set_title("Search statistic over the 2D parameter box"); fig.tight_layout(); fig.savefig(f"{FIG}/landscape.png"); plt.close()

# %% [markdown]
# ## 3. PSO from scratch
# %%
def pso(fun, lb, ub, n=25, iters=100, w=0.9, c1=0.4, c2=0.4, topology="gbest", k=5,
        seed=0, ftol=None, ftol_iter=10):
    rng = np.random.default_rng(seed); D = len(lb)
    x = rng.uniform(lb, ub, (n, D)); v = np.zeros((n, D))
    fx = np.array([fun(p) for p in x]); pb, pf = x.copy(), fx.copy()
    hist = dict(pos=[x.copy()], best=[pf.min()], nev=n)
    stall = 0
    for it in range(iters):
        if topology == "gbest":
            g = np.tile(pb[pf.argmin()], (n, 1))
        else:                                          # ring-like: k nearest neighbours (raw L2, as pyswarms)
            dist = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
            nb = np.argsort(dist, axis=1)[:, :k]
            g = pb[nb[np.arange(n), pf[nb].argmin(axis=1)]]
        r1, r2 = rng.random((n, D)), rng.random((n, D))
        v = w*v + c1*r1*(pb - x) + c2*r2*(g - x)
        x = x + v
        out = (x < lb) | (x > ub); x = np.clip(x, lb, ub); v[out] = 0.0
        fx = np.array([fun(p) for p in x]); hist["nev"] += n
        imp = fx < pf; pb[imp], pf[imp] = x[imp], fx[imp]
        hist["pos"].append(x.copy()); hist["best"].append(pf.min())
        if ftol is not None:
            stall = stall + 1 if abs(hist["best"][-2] - hist["best"][-1]) < ftol else 0
            if stall >= ftol_iter: break
    j = pf.argmin(); hist["pos"] = np.array(hist["pos"]); hist["best"] = np.array(hist["best"])
    return pb[j], pf[j], hist

# %% [markdown]
# ## 4. One run, drawn on the landscape
# %%
fun = lambda p: cost(p, d)
xb, fb, h = pso(fun, lb, ub, seed=3, iters=100)
print("single run: Mc=%.3f eta=%.4f rho=%.2f, evals=%d" % (xb[0], xb[1], -fb, h["nev"]))
fig, ax = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
for a, it in zip(ax, [0, 10, 100]):
    a.contourf(Mcs, etas, L, 30, cmap="viridis")
    a.plot(h["pos"][it][:, 0], h["pos"][it][:, 1], "w.", ms=7); a.plot(*theta_t, "r*", ms=11)
    a.set_title(f"iteration {it}"); a.set_xlabel(r"$\mathcal{M}_c$")
ax[0].set_ylabel(r"$\eta$"); fig.tight_layout(); fig.savefig(f"{FIG}/swarm_snapshots.png"); plt.close()

# %% [markdown]
# ## 5. Settings matter: your options (w=0.9, c1=c2=0.4) vs textbook (w=0.7, c=1.5)
# %%
cfgs = {"your settings, ring k=5": dict(w=0.9, c1=0.4, c2=0.4, topology="ring"),
        "your settings, global best": dict(w=0.9, c1=0.4, c2=0.4, topology="gbest"),
        "textbook, global best": dict(w=0.7, c1=1.5, c2=1.5, topology="gbest")}
res = {}
fig, ax = plt.subplots(figsize=(5.6, 4))
for name, kw in cfgs.items():
    runs = [pso(fun, lb, ub, seed=s, iters=100, **kw) for s in range(12)]
    B = np.array([-r[2]["best"] for r in runs]); res[name] = runs
    ax.plot(np.median(B, 0), label=name); ax.fill_between(range(B.shape[1]), B.min(0), B.max(0), alpha=0.15)
    fin = np.array([r[0] for r in runs])
    print("%-28s median final rho %.2f | min %.2f | final Mc spread (std) %.3f" % (name, np.median(B[:, -1]), B[:, -1].min(), fin[:, 0].std()))
ax.axhline(snr_series(d, template(*theta_t)).max(), color="k", ls=":", label="rho at truth")
ax.set_xlabel("iteration"); ax.set_ylabel("best rho so far"); ax.legend(fontsize=8); ax.set_title("Convergence, 12 seeds each (band = min/max)")
fig.tight_layout(); fig.savefig(f"{FIG}/convergence.png"); plt.close()

# %% [markdown]
# ## 6. Independent runs and swarm spread: why spread is NOT an error bar
# %%
runs = res["textbook, global best"]
fin = np.array([r[0] for r in runs])
spread = np.array([[np.std(p[:, 0]) for p in r[2]["pos"]] for r in runs])
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].contourf(Mcs, etas, L, 30, cmap="viridis"); ax[0].plot(fin[:, 0], fin[:, 1], "wo", ms=5, mec="k"); ax[0].plot(*theta_t, "r*", ms=12)
ax[0].set_xlabel(r"$\mathcal{M}_c$"); ax[0].set_ylabel(r"$\eta$"); ax[0].set_title("final point of 12 independent runs")
ax[1].semilogy(spread.T, color="C0", alpha=0.5); ax[1].set_xlabel("iteration"); ax[1].set_ylabel(r"std of swarm in $\mathcal{M}_c$"); ax[1].set_title("swarm spread collapses by construction")
fig.tight_layout(); fig.savefig(f"{FIG}/spread.png"); plt.close()
print("final-point scatter across runs: std Mc=%.3f, std eta=%.4f" % tuple(fin.std(0)))
print("swarm std in Mc: start %.2f -> end (median) %.4f" % (spread[:, 0].mean(), np.median(spread[:, -1])))

# %% [markdown]
# ## 7. Signal vs noise-only: the idea behind foreground/background
# %%
nrun = int(os.environ.get("NNOISE", 40))
rs, rn = [], []
for s in range(nrun):
    ds = make_data(theta_t, 12.0, seed=100+s)
    dn = make_data(theta_t, 0.0, seed=500+s, signal=False)
    rs.append(-pso(lambda p: cost(p, ds), lb, ub, w=0.7, c1=1.5, c2=1.5, seed=s, iters=60)[1])
    rn.append(-pso(lambda p: cost(p, dn), lb, ub, w=0.7, c1=1.5, c2=1.5, seed=s, iters=60)[1])
rs, rn = np.array(rs), np.array(rn)
print("noise-only: PSO best rho median %.2f, max %.2f | injected (SNR 12): median %.2f, min %.2f" % (np.median(rn), rn.max(), np.median(rs), rs.min()))
fig, ax = plt.subplots(figsize=(5.6, 3.8)); bins = np.linspace(3, 16, 27)
ax.hist(rn, bins, alpha=0.6, label="noise only"); ax.hist(rs, bins, alpha=0.6, label="signal injected (optimal SNR 12)")
ax.set_xlabel(r"best $\rho$ found by PSO"); ax.set_ylabel("runs"); ax.legend(); fig.tight_layout(); fig.savefig(f"{FIG}/signal_vs_noise.png"); plt.close()
print("DONE")

# %% [markdown]
# ## 8. The honest error bar: repeat the experiment over NOISE realisations
# One PSO run finds the maximum-likelihood point of ONE noise realisation. How far that point sits from the truth
# is the statistical error. It must be measured over many noise realisations, not read off the swarm.
# %%
nrep = int(os.environ.get("NREP", 30))
est = []
for s in range(nrep):
    ds = make_data(theta_t, 20.0, seed=2000+s)
    xb_, fb_, _ = pso(lambda p: cost(p, ds), lb, ub, w=0.7, c1=1.5, c2=1.5, seed=s, iters=60)
    est.append(xb_)
est = np.array(est)
print("over %d noise realisations: Mc* = %.3f +- %.3f (truth %.3f); eta* = %.3f +- %.3f (truth %.3f)" % (
    nrep, est[:,0].mean(), est[:,0].std(), theta_t[0], est[:,1].mean(), est[:,1].std(), theta_t[1]))
print("fraction of runs pinned at the eta upper bound 0.25: %.2f" % np.mean(est[:,1] > 0.2499))
np.save("theta_star_noise_realisations.npy", est)
fig, ax = plt.subplots(figsize=(5.6, 4))
ax.contourf(Mcs, etas, L, 30, cmap="viridis"); ax.plot(est[:,0], est[:,1], "wo", mec="k", ms=5, label="PSO optimum, one point per noise realisation")
ax.plot(*theta_t, "r*", ms=13, label="truth"); ax.set_xlabel(r"$\mathcal{M}_c$"); ax.set_ylabel(r"$\eta$"); ax.legend(fontsize=7, loc="lower left")
ax.set_title("Spread of the optimum over noise realisations"); fig.tight_layout(); fig.savefig(f"{FIG}/noise_scatter.png"); plt.close()
