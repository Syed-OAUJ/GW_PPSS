# Project README: Testing the Spin-Induced Quadrupole Moment (κ) with a PSO Non-Template Search

Welcome. This file explains what we are doing, why, and what you will do each month. Read it first, then the documents listed at the end.

## 1. The one-paragraph version

Every compact object has a spin-induced quadrupole moment, κ. For a black hole, κ = 1. For other objects, such as boson stars or neutron stars, κ is different. κ leaves a small imprint on the gravitational-wave (GW) phase of an inspiralling binary, so measuring it tests whether the objects are black holes. We want to know two things. First, how well can κ be measured (statistical error)? Second, what goes wrong if the waveform model is wrong, for example if we ignore orbital eccentricity (systematic bias)? To explore this we use a **particle swarm optimiser (PSO)** as a search tool that does not need a pre-built template bank.

## 2. Background you need

| Idea | In one line |
|---|---|
| κ (SIQM) | Spin-induced quadrupole moment. κ = 1 for black holes. It enters the GW phase at 2PN and 3PN (Krishnendu, Arun & Mishra 2017, arXiv:1701.06318). |
| Matched filter | Slide a template against the data and compute the signal-to-noise ratio (SNR). The inner product is (a\|b) = 4 Re ∫ a b* / Sn df. |
| Fisher matrix | Fast estimate of the statistical error from the curvature of the likelihood around the true parameters. Valid at high SNR. |
| Bayesian analysis | Full posterior over the parameters. It gives honest uncertainties and handles degeneracies. |
| Eccentricity e0 | Orbits that are not perfectly circular. If we fit with a circular model, κ can be biased. |
| PSO | A swarm of candidate parameter sets moves through parameter space towards the highest SNR. It gives a **point estimate**, not an uncertainty. |

## 3. The key idea: two pillars

1. **Pillar 1: statistical error (Fisher vs Bayesian).** You have already done this with a simple 2PN κ waveform for boson stars. The next step is the full κ waveform (2PN and 3PN terms).
2. **Pillar 2: PSO non-template search.** Search with a circular model, and again with an eccentric model, and measure how far the best-fit κ moves. The bias is quantified as **R(e0) = |bias| / σ_κ**. If R > 1, the bias is larger than the statistical error and the circular model is not safe.

Pillar 1 gives σ_κ. Pillar 2 gives the bias. The ratio of the two is the main result.

## 4. The most important thing to understand about PSO

**A converged swarm collapses onto one point. Its spread is NOT an error bar.**

In our toy test (SNR 20) 12 independent runs on the same data ended at the same point, and the swarm width shrank to about 3×10⁻⁴ M☉. The real statistical scatter of the optimum over different noise realisations was about 0.013 M☉, roughly 40 times larger. So:

- PSO gives the best-fit point and a check for multiple peaks.
- Fisher and Bayesian analyses give the uncertainty.
- Never quote swarm spread as a measurement error.

## 5. Workflow

```
 Fisher (circular κ)  ──►  σ_κ
                                   ╲
 PSO circular search  ──►  κ̂_circ   ──►  bias, R(e0) = |bias| / σ_κ
 PSO eccentric search ──►  κ̂_ecc   ╱
                                   ──►  Bayesian check on a few cases
```

## 6. Timeline (about 7 months)

Details are in `roadmap.pdf`. In short:

- **Months 1–2:** run the toy PSO example, understand the real code, and set up waveforms and PSDs.
- **Months 3–4:** Fisher analysis with the full κ waveform. Check against the Bayesian posterior.
- **Months 5–6:** PSO circular vs eccentric searches. Compute R(e0).
- **Month 7:** write up, make figures, and prepare the paper draft.

## 7. What is out of scope

To keep this finishable, we are **not** doing: real GW241011 data, Einstein Telescope mock data, rotating neutron stars, or a full multi-detector PyCBC network code.

## 8. How to start (first week)

1. Run `pso_gw_toy.py` (it takes several minutes; reduce `NREP` for a faster run).
2. Read the guide `PSO_how_the_code_runs.pdf`, Part A first, then Part B.
3. Answer the exercises at the end of Part A in your own words.
4. Open the real code `global_optimizer_inspiral.py` and find where each toy step lives.

## 9. Ground rules

- Use **geometrised units** carefully. Convert masses to seconds with G·M☉/c³ = 4.925491×10⁻⁶ s. Mistakes here silently break the physics.
- Always use **raw SNR** (not the reweighted statistic) for fitting factors.
- Label every result with its noise realisation, SNR, and settings.
- If a result looks too good, check it. Ask questions early.
- Keep a short log of what you ran and what you found.

## 10. Known open items (do not assume these are settled)

- The 3PN κ coefficient must be transcribed from arXiv:1701.06318 (Eq. 1.5b) and the 2PN coefficient verified.
- In the real code, check three things: whether `out_vals['stat'].max()` fails on an empty array, whether the swarm is re-initialised per segment, and whether `pyswarms` neighbourhood distances use raw units.
- The toy PSD is an analytic placeholder, to be replaced by real aLIGO, ET and CE curves.
- A literature check is needed to confirm no prior work combines κ and eccentricity in a PSO search.

## 11. Documents

| File | Purpose |
|---|---|
| `Pipeline_latex.pdf` | Full pipeline description (Fisher, Bayes, PSO) |
| `roadmap.pdf` | Month-by-month plan |
| `pso_vs_bayes_lecture.pdf` | Short lecture: what PSO can and cannot tell us |
| `PSO_how_the_code_runs.pdf` | Toy test case plus the map to the real code |
| `pso_gw_toy.py` | Runnable toy example |

## 12. Key references

- Krishnendu, Arun & Mishra 2017, arXiv:1701.06318 (κ in the GW phase)
- Naqvi & Mishra 2025, arXiv:2509.10675
- Krishnendu 2025, arXiv:2512.20579
- Chia et al. 2022, arXiv:2211.00039 (κ_eff reduced waveforms)
- Pal & Nayak 2023, arXiv:2307.03736 (PSO for eccentric searches)

Questions go to Umair (IFJ PAN). Bring a short written summary of what you ran and found to each weekly meeting.
