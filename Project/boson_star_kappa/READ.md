# boson_star_kappa

Fisher-matrix and Bayesian (MCMC) forecasts of how well gravitational-wave inspiral signals can constrain the spin-induced quadrupole parameter **κ_s** of compact binaries (boson stars vs. black holes). Uses the 3.5PN spin-aligned phasing with quadrupole-monopole terms (AISS), and recreates results from Krishnendu, Mishra, Arun (arXiv:1811.00317).

Parameter vector: `theta = (t_c, phi_c, M_c, eta, kappa_s)`. Fixed inputs: `chi1`, `chi2`, `kappa_a`.

## Folder layout

```
boson_star_kappa/
├── run.py               # entry point (the only file you run)
├── config.py            # all constants, fiducial values, paths, settings
├── waveform.py          # symbolic PN phasing -> fast numpy waveform
├── psd.py               # detector noise curves, SNR integrals
├── boson_star_data.py   # boson-star (mass, spin, kappa) table loading
├── fisher.py            # Fisher matrix, covariance, error tables
├── bayes.py             # likelihood, prior, emcee sampler
├── plots.py             # every figure
├── data/
│   └── solitonic_sigma_0.04.dat
├── Numerical_PSD/
│   └── aLIGO_ZeroDetHighPower_psd.txt
└── output/              # created automatically; all results go here
```

## Requirements

Python 3.9+ and:

```
pip install sympy numpy scipy pandas matplotlib emcee corner
```

PyCBC is only needed for `psd-gw150914` (`pip install pycbc`).

## Running

From the project folder:

```
python3 run.py <command>
```

| Command | What it does | Output (in `output/`) |
|---|---|---|
| `fisher-table` | Errors on all parameters vs. PN order (1PN–3.5PN) for the boson-star pair | `TableRecreated.csv`, `kappa_vs_mass.png` |
| `fig2` | Recreates Fig. 2: 1σ error on κ_s vs. total mass, for several mass ratios and spin configurations | `Fig2_recreated.png`, `Fig2_data.csv` |
| `mcmc` | Runs emcee on a zero-noise injection and makes a corner plot | `mcmc_samples.npy`, `kappa_corner.png` |
| `compare` | Overlays Fisher Gaussian ellipses on the MCMC corner plot (M_c, eta, kappa_s) and prints a comparison table | `fisher_vs_bayes_corner.png` |
| `psd-fits` | Compares the analytical aLIGO fits with the numerical PSD; plots the SNR density | `aligo_psd_comparison.png`, `SNR_density.png` |
| `psd-gw150914` | Analytical vs. measured GW150914 PSDs (needs PyCBC and internet) | `PSD_plot.png` |

**Order matters:** run `mcmc` before `compare`, since `compare` reads `output/mcmc_samples.npy`.

### In PyCharm

Open the folder as the project, then Run → Edit Configurations → **+** → Python. Set *Script path* to `run.py`, *Parameters* to a command (e.g. `fisher-table`) and *Working directory* to the project folder. Make one configuration per command.

## Changing settings

Everything adjustable lives in `config.py`:

- **Fiducial source:** `CHI1`, `CHI2`, `BOSON_STAR_ROWS` (which table rows give bodies 1 and 2), `KAPPA_S_BH`, `KAPPA_A_BH`
- **Fisher:** `MAX_PN_ORDER` (7 = 3.5PN), `SNR_FIDUCIAL`, `F_LOW`, `N_FREQS_FISHER`
- **MCMC:** `N_WALKERS`, `N_STEPS`, `BURN_IN`, `THIN`, `ADD_NOISE`, `PRIOR`
- **Fig. 2:** `FIG2_TOTAL_MASSES`, `FIG2_TOP_CASES`, `FIG2_BOTTOM_CASES`
- **Paths:** `NUMERICAL_PSD_FILE`, `BOSON_STAR_FILE`

## Using the modules from Python

```python
import fisher, waveform as wf

# 1-sigma error on kappa_s: total mass 30 Msun, q = 1.2, chi = (0.9, 0.8), BBH fiducial
print(fisher.delta_kappa_s(30.0, 1.2, 0.9, 0.8))

# full covariance for a custom system (masses in seconds, G = c = 1)
M, eta, Mc = wf.total_mass_q_to_system(30.0, 1.2)
cov = fisher.fisher_covariance(Mc, eta, kappa_s=1.0, kappa_a=0.0, chi1=0.9, chi2=0.8)
```

```python
import bayes, waveform as wf

M, eta, Mc = wf.total_mass_q_to_system(30.0, 1.2)
model = bayes.BayesModel(Mc, eta, kappa_s=1.0, kappa_a=0.0, chi1=0.9, chi2=0.8)
sampler = bayes.run_mcmc(model)
samples = bayes.flatten_chain(sampler)
```

## Conventions

- Units: G = c = 1, masses in **seconds** (`config.MSUN_SEC` converts from solar masses).
- Frequency grids are log-spaced from `F_LOW` (20 Hz) to the Schwarzschild LSO frequency.
- `kappa_s = (kappa1 + kappa2)/2`, `kappa_a = (kappa1 - kappa2)/2`. Only `kappa_s` is varied; `kappa_a`, `chi1`, `chi2` are fixed.
- κ_s first enters the phase at 2PN, so Δκ_s is `NaN` for `max_order < 4`.
- Fisher matrix: `Gamma_ij = 4 A² ∫ f^(-7/3) ∂_i Φ ∂_j Φ / S_h(f) df`, with amplitude `A` set by the fiducial SNR.
- The Fisher code raises an error at exactly `eta = 0.25` with `chi1 != chi2` (derivative diverges); use slightly unequal masses.

## Troubleshooting

- **`error: the following arguments are required: command`**: pass a command, e.g. `python3 run.py fig2`.
- **`FileNotFoundError` for a `.dat` / `.txt` file**: keep `data/` and `Numerical_PSD/` next to `run.py`, or edit the paths in `config.py`.
- **`ModuleNotFoundError`**: install the missing package into the interpreter that runs the script.
- **`compare` fails on `mcmc_samples.npy`**: run `mcmc` first.
