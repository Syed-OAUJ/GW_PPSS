# Independent Research Project: Constraining Exotic Compact Objects with Gravitational Waves

**From Boson Stars to Einstein Telescope**

---

## Project Overview

The direct detection of gravitational waves has opened a new avenue for probing the nature of compact objects in the strong-gravity regime. While current observations are consistent with the Kerr black-hole paradigm, theoretical models suggest the existence of exotic compact objects (ECOs) such as boson stars and other horizonless alternatives. A key diagnostic is the measurement of spin-induced multipole moments, which are sensitive to the internal mass and current distribution of a compact object. Specifically, the spin-induced quadrupole moment serves as an important probe of deviations from the Kerr metric, allowing us to constrain the structure of these objects beyond the black-hole hypothesis.

This project focuses on developing parameter estimation techniques to analyze these observables. Students will use both Python and Mathematica to implement Fisher information matrices — a method for estimating the precision of parameter measurements — and Bayesian inference, which provides a probabilistic framework for model comparison. By analyzing simulated compact-binary signals, participants will investigate how waveform modelling assumptions, particularly orbital eccentricity, affect our ability to resolve these moments.

The project concludes by assessing the robustness of these tests for next-generation observatories like the Einstein Telescope (ET), which will feature a triangular configuration and significantly improved low-frequency sensitivity, allowing observation of more inspiral cycles and thus extraction of multipole moments with much greater precision.

*This project directly builds on published research: [arXiv:2511.17341](https://arxiv.org/abs/2511.17341), "Implications of GW241011 for rotating exotic compact objects."*

---

## Papers (`/Papers` folder)

Two background papers to read alongside the lectures, in this order:

1. **[arXiv:1602.03840](https://arxiv.org/pdf/1602.03840)** — "Observation of Gravitational Waves from a Binary Black Hole Merger" (Abbott et al. 2016). The GW150914 discovery paper. Read this first, to see what a real detection actually looks like end to end — the signal, the detectors, the parameter estimates — before touching any of the machinery that produces those estimates.

2. **[arXiv:gr-qc/0411146](https://arxiv.org/abs/gr-qc/0411146)** — Arun, Iyer, Sathyaprakash & Sundararajan, "Parameter estimation of inspiralling compact binaries using 3.5PN gravitational wave phasing: the non-spinning case." *(To be added to the repo.)* This is a Fisher-matrix forecasting paper — Table 1 reports predicted parameter uncertainties ($\Delta\mathcal{M}$, $\Delta\eta$, etc.) at various post-Newtonian orders for representative binaries. **Once you have working Fisher-matrix code (Week 3), try to reproduce a row of Table 1 for one of their example systems as a validation check** — if your code roughly matches a published, peer-reviewed result, that's strong evidence it's actually correct, not just "runs without crashing."

---

## Week 1: Lectures (July 6-10) — shared with full cohort

### Lecture 1: Gravitational Wave Basics & Compact Binaries
* The quadrupole formula: $h_{ij} = \frac{2G}{c^4 D} \ddot{I}_{ij}$.
* Chirp waveform: $h(t) \propto \cos(\Phi(t))$, $f(t) = \frac{1}{\pi}\frac{d\Phi}{dt}$.
* LIGO/Virgo/KAGRA sensitivity curves and detections (GWTC series).
* Binary black holes vs. neutron stars vs. exotic compact objects (ECOs).

### Lecture 2: Exotic Compact Objects and Particle Physics
* Boson stars: complex scalar field $\phi$ with potential $V(|\phi|^2)$.
* Repulsive potential: $V_{\text{rep}} = \mu^2|\phi|^2 + \lambda|\phi|^4$.
* Solitonic and axionic potentials (see [arXiv:2511.17341](https://arxiv.org/abs/2511.17341)).
* Connection to dark matter: self-interacting dark matter cross sections.

### Lecture 3: Spin-Induced Quadrupole Moment (SIQM)
* For Kerr BH: $Q = -\chi^2 M^3$, i.e. reduced quadrupole $\kappa = 1$.
* For ECOs: $\kappa \neq 1$ (can be >1 or <1 depending on compactness).
* Effect on inspiral phase: at 2PN and 3PN order (Eqs. 1.5a,b of [arXiv:1701.06318](https://arxiv.org/abs/1701.06318)).
* Parametrisation: $Q_i = -\kappa_i \chi_i^2 m_i^3$, with $\kappa_i$ introduced in the waveform.

### Lecture 4: Bayesian Inference for Gravitational Waves
* Bayes' theorem: $p(\vec{\theta}|d) = \frac{\pi(\vec{\theta})\mathcal{L}(d|\vec{\theta})}{\mathcal{Z}}$.
* Likelihood for stationary Gaussian noise: $\ln\mathcal{L} \propto -\frac{1}{2}\langle d-h|d-h\rangle$.
* Nested sampling (Dynesty) vs. MCMC.
* Example: posterior for $\kappa_1$ from GW241011 (Fig. 1 in the paper).

### Lecture 5: Fisher Matrix and Forecasting for Future Detectors
* Fisher information matrix: $\Gamma_{ij} = \left\langle \frac{\partial h}{\partial \theta_i} \middle| \frac{\partial h}{\partial \theta_j} \right\rangle$.
* Cramér–Rao bound: $\sigma_{\theta_i} \ge \sqrt{(\Gamma^{-1})_{ii}}$.
* Einstein Telescope (ET): design sensitivity, $f_{\text{min}} = 1$ Hz, $f_{\text{max}} = 10^4$ Hz.
* Forecast for $\kappa_1$: expected $\sigma_{\kappa_1} \approx 0.1$ at SNR=100.

### Daily Practicals (1-2h each)
* **Monday:** Python crash course (numpy, matplotlib). *Covered by the general Workshop / Parts I-III of the Colab notebook.*
* **Tuesday:** Reading public GW data (GWOSC) and plotting strain.
* **Wednesday:** Simple frequency-domain inspiral waveform generator. *Covered by Part I of the Colab notebook — can be reused directly.*
* **Thursday:** Introduction to `bilby` (parameter estimation).
* **Friday:** Coding a Fisher matrix from scratch. *Covered by Part VI of the Colab notebook — can be reused directly.*

> **Note:** Tuesday and Thursday need dedicated material not covered by the general workshop notebook (GWOSC data access, and a `bilby` intro respectively) — flag if you'd like help building either.

---

## Weeks 2-4: Two Parallel Tracks

Every week runs two tracks side by side, both answering the same physics question:

- **Core track** (both students, together): a small, controlled extension of the general workshop's toy machinery. Low risk, guaranteed to produce a clean, presentable result by Week 4.
- **Stretch track** (for future): the same question using real infrastructure (`pycbc` + `TaylorF2`/`dQuadMon`, real detector PSDs). This is what starts becoming genuinely pipeline-shaped (potentially aim for paper).

Noise is handled with real detector data throughout — aLIGO, ET, and CE PSD text files (already available) rather than the flat-noise simplification used in the general workshop.

### Week 2: Waveform Simulation & Injection

**Teach:** the $\kappa$-dependent 2PN SIQM phase term (aligned-spin, fix $\kappa_2=1$); how to go from flat-noise to a real noise-weighted likelihood by loading a PSD file and replacing $1/\sigma^2$ with $1/S_n(f)$.

**Core track:**
- Extend the toy `waveform()` with a single $\kappa_1$ phase term.
- Load the aLIGO PSD file, interpolate **in log-log space** (PSDs span many orders of magnitude — linear interpolation distorts them) onto the toy model's frequency grid.
- Compute a real matched-filter SNR using the actual noise-weighted inner product from Lecture 1.
- **Deliverable:** dephasing/mismatch vs. $\kappa_1$ plot, with a real (PSD-weighted) SNR attached.

**Stretch track:**
- Same physics via `pycbc.waveform.get_fd_waveform(approximant='TaylorF2', dquad_mon1=kappa1-1, ...)`.
- Load the same PSD file with `pycbc.psd.read.from_txt(...)`.
- **Deliverable:** same dephasing/mismatch plot, real waveform. Compare directly against the core track's plot — same qualitative trend, different fidelity; a natural way to satisfy the programme's "students cooperate" expectation.

### Week 3: Fisher Matrix Analysis

**Teach:** swapping $1/\sigma^2 \to 1/S_n(f)$ inside `fisher_matrix()` is a one-line change from the workshop's version; comparing across detectors (aLIGO vs. ET vs. CE) is the natural next question once all three PSDs are loaded — this is your paper's own motivation, scaled to project size.

**Core track:**
- 4-parameter Fisher matrix $\{\ln M_c, \kappa_1, t_c, \phi_c\}$, PSD-weighted, computed separately for aLIGO, ET, and CE.
- **Deliverable:** $\sigma_{\kappa_1}$ vs. SNR (or vs. detector), one line per detector — a clean, guaranteed plot showing the ET/CE improvement quantitatively.
- **Validation check:** try to reproduce a row of Table 1 from arXiv:gr-qc/0411146 with your own code (see Papers section above).

**Stretch track:**
- Full 8-parameter Fisher matrix on the `pycbc`/`TaylorF2` waveform, same three detectors.
- Expect real numerical fragility — re-run the condition-number check from the workshop before trusting any 8x8 result; budget a session for exactly this kind of debugging.

### Week 4: Bayesian Recovery

**Teach:** reading a $\kappa_1$ posterior against $\kappa=1$ (Kerr) — does it exclude Kerr at some credible level, and does that change across detectors?

**Core track:**
- `emcee` on the same 4-parameter, PSD-weighted model — one run per detector if time allows (aLIGO vs. ET is the more dramatic contrast; CE optional if time is short).
- **Deliverable (guaranteed centerpiece):** corner plot(s) showing the $\kappa_1$ posterior tightening from aLIGO to ET.

**Stretch track:**
- `bilby` + `dynesty` on the full `pycbc` model, submitted **early in the week**, not during the final session — full nested sampling on an 8-parameter waveform can take many hours. If only one detector's run finishes in time, make it ET. Frame explicitly as pipeline groundwork for a future paper, not a finished result.

---

## Suggested Final Presentation

1. Motivation (your paper — GW241011, the ECO question)
2. Toy pipeline overview + dephasing plot (Week 2 core)
3. **$\sigma_{\kappa_1}$ vs. detector (aLIGO / ET / CE)** (Week 3/4 core) — the guaranteed centerpiece, directly echoing your paper's own question about next-generation detector capability
4. Bayesian recovery corner plot(s), aLIGO vs. ET (Week 4 core)
5. *(If ready)* Stretch track snapshot: real waveform, real PSD, preliminary `bilby` status — framed as next steps toward publication

---

## Appendix: Data Source

Spin-induced quadrupole moments for boson stars: [github.com/tamaraevst/Spin-induced-quadrupole-moments-of-boson-stars](https://github.com/tamaraevst/Spin-induced-quadrupole-moments-of-boson-stars)
