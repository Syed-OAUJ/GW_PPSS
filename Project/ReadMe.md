# Independent Research Project: Constraining Exotic Compact Objects with Gravitational Waves

**From Boson Stars to Einstein Telescope**


---

## Project Overview

The direct detection of gravitational waves has opened a new avenue for probing the nature of compact objects in the strong-gravity regime. While current observations are consistent with the Kerr black-hole paradigm, theoretical models suggest the existence of exotic compact objects (ECOs) such as boson stars and other horizonless alternatives. A key diagnostic is the measurement of spin-induced multipole moments, which are sensitive to the internal mass and current distribution of a compact object. Specifically, the spin-induced quadrupole moment serves as an important probe of deviations from the Kerr metric, allowing us to constrain the structure of these objects beyond the black-hole hypothesis.

This project focuses on developing parameter estimation techniques to analyze these observables. Students will use both Python and Mathematica to implement Fisher information matrices — a method for estimating the precision of parameter measurements — and Bayesian inference, which provides a probabilistic framework for model comparison. By analyzing simulated compact-binary signals, participants will investigate how waveform modelling assumptions, particularly orbital eccentricity, affect our ability to resolve these moments.

The project concludes by assessing the robustness of these tests for next-generation observatories like the Einstein Telescope (ET), which will feature a triangular configuration and significantly improved low-frequency sensitivity, allowing observation of more inspiral cycles and thus extraction of multipole moments with much greater precision.

*This project directly builds on published research: [arXiv:2511.17341](https://arxiv.org/abs/2511.17341), "Implications of GW241011 for rotating exotic compact objects."*

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
* **Monday:** Python crash course (numpy, matplotlib). *Covered by the general [Workshop](./Home) / Parts I-III of the Colab notebook.*
* **Tuesday:** Reading public GW data (GWOSC) and plotting strain.
* **Wednesday:** Simple frequency-domain inspiral waveform generator. *Covered by Part I of the Colab notebook — can be reused directly.*
* **Thursday:** Introduction to `bilby` (parameter estimation).
* **Friday:** Coding a Fisher matrix from scratch. *Covered by Part VI of the Colab notebook — can be reused directly.*

> **Note:** Tuesday and Thursday need dedicated material not covered by the general workshop notebook (GWOSC data access, and a `bilby` intro respectively) — flag if you'd like help building either.

---

## Week 2: Waveform Simulation & Injection

Students implement a restricted PN waveform with $\kappa_1$ dependence, inject signals into simulated ET noise, and compare Kerr vs. ECO waveforms.

> **Note:** there's a real modeling gap between the general workshop's toy waveform (non-spinning, no $\kappa$ term) and this week's task. Consider a short bridging exercise first — extending the toy notebook's phase formula with a placeholder $\kappa$-dependent term — before moving to the full 2PN/3PN implementation.

## Week 3: Fisher Matrix Analysis

Compute the Fisher matrix for parameters $\{\ln\mathcal{A}, \ln M_c, \eta, \chi_1, \chi_2, \kappa_1, t_c, \phi_c\}$; obtain $\sigma_{\kappa_1}$ as a function of SNR.

> **Note:** the `fisher_matrix()` function from the general workshop notebook generalizes cleanly to N parameters (it already loops over `ndim`), so the method transfers directly — but re-run the condition-number sanity check from that notebook on the full 8x8 matrix before trusting the result; finite-difference Fisher matrices get numerically fragile fast in higher dimensions.

## Week 4: Bayesian Recovery with `bilby`

Run full nested sampling on simulated ET data, produce corner plots, and write a short report. Discuss which ECO models (repulsive, solitonic, axionic) can be distinguished from black holes.

> **Note:** full `dynesty` nested sampling on an 8-parameter waveform can take many hours. Submit these jobs well before this week's session starts, and consider high-SNR zero-noise injections plus a reduced live-point count to keep runtimes manageable within the week.

---

## Appendix: Data Source

Spin-induced quadrupole moments for boson stars: [github.com/tamaraevst/Spin-induced-quadrupole-moments-of-boson-stars](https://github.com/tamaraevst/Spin-induced-quadrupole-moments-of-boson-stars)
