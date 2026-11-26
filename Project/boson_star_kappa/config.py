"""
config.py -- every constant, fiducial value, path and run setting in one place.

Nothing in here imports project code, so any other module can import it safely.
Edit values here; do not hard-code numbers in the logic modules.
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

NUMERICAL_PSD_FILE = ROOT /"data" / "Numerical_PSD" / "aLIGO_ZeroDetHighPower_psd.txt"
BOSON_STAR_FILE = ROOT / "data" / "solitonic_sigma_0.04.dat"
BOSON_STAR_COLUMNS = ["freq", "mass", "ang", "radius99", "radius95", "compactness", "kappa"]

MCMC_SAMPLES_FILE = OUTPUT_DIR / "mcmc_samples.npy"

# ----------------------------------------------------------------------------
# Units and PN constants (G = c = 1, masses in seconds)
# ----------------------------------------------------------------------------
MSUN_SEC = 4.925491025543576e-6      # G*M_sun/c^3 [s]
GAMMA_E = 0.5772156649015329         # Euler-Mascheroni constant
THETA_PN = -1.28                     # 3PN ambiguity parameter (theta)
LAMBDA_PN = -0.6451                  # 3PN ambiguity parameter (lambda)
F_REF_LOG = 10.0                     # [Hz] reference frequency inside vlso of the 2.5PN log term

# ----------------------------------------------------------------------------
# Detector noise
# ----------------------------------------------------------------------------
# Eq. (3.7a) fit, and the older fit used in Refs. [6,22,25,46]
ALIGO_NEW_FIT = dict(f0=215.0, fs=20.0, S0=1e-49)
ALIGO_OLD_FIT = dict(f0=70.0, fs=10.0, S0=6e-49)

F_LOW = 20.0                         # [Hz] low-frequency cutoff for SNR / Fisher integrals
F_PLOT_RANGE = (10, 2000)

# ----------------------------------------------------------------------------
# Fisher-matrix settings
# ----------------------------------------------------------------------------
MAX_PN_ORDER = 7                     # 7 -> 3.5PN
SNR_FIDUCIAL = 10.0
N_FREQS_FISHER = 500
PN_LABELS = {2: "1PN", 3: "1.5PN", 4: "2PN", 5: "2.5PN", 6: "3PN", 7: "3.5PN"}

# parameter vector theta = (t_c, phi_c, M_c, eta, kappa_s)
PARAM_LABELS = [r"$t_c$", r"$\phi_c$", r"$M_c$", r"$\eta$", r"$\kappa_s$"]
KAPPA_S_INDEX = 4

# ----------------------------------------------------------------------------
# Fiducial source
# ----------------------------------------------------------------------------
CHI1, CHI2 = 0.9, 0.8
T_C, PHI_C = 0.0, 0.0
BOSON_STAR_ROWS = (0, 1)             # rows of BOSON_STAR_FILE used for body 1 / body 2
KAPPA_S_BH, KAPPA_A_BH = 1.0, 0.0    # binary black hole reference

# ----------------------------------------------------------------------------
# Bayesian / MCMC settings
# ----------------------------------------------------------------------------
N_FREQS_MCMC = 2000
ADD_NOISE = False                    # zero-noise injection by default
NOISE_SEED = 42
INIT_SEED = 7
N_WALKERS, N_STEPS = 32, 3000
BURN_IN, THIN = 500, 10
INIT_SPREAD = 0.5                    # walker ball radius in units of the Fisher sigma

PRIOR = dict(
    t_c=(-0.1, 0.1),
    phi_c=(-3.141592653589793, 3.141592653589793),
    Mc_fraction=(0.5, 1.5),          # prior on M_c as a fraction of the injected M_c
    eta=(0.05, 0.25),
    kappa_s=(0.0, 10.0),
)
FALLBACK_SCALE = dict(t_c=1e-4, phi_c=0.1, Mc_fraction=0.05, eta=0.02, kappa_s=0.2)

# ----------------------------------------------------------------------------
# Fig. 2 of Krishnendu, Mishra, Arun (arXiv:1811.00317)
# ----------------------------------------------------------------------------
FIG2_TOTAL_MASSES = list(range(10, 81, 10))   # [Msun]
FIG2_TOP_CASES = [                             # (q, label, color, marker); spins = (CHI1, CHI2)
    (1.2, "q=1.2", "tab:orange", "D"),
    (3.0, "q=3", "gray", "s"),
    (5.0, "q=5", "black", "v"),
]
FIG2_BOTTOM_Q = 1.2
FIG2_BOTTOM_CASES = [                          # (chi1, chi2, color, marker)
    (0.9, 0.8, "tab:orange", "D"),
    (0.9, -0.8, "gray", "s"),
    (-0.9, 0.8, "black", "v"),
    (-0.9, -0.8, "olive", "^"),
]

# ----------------------------------------------------------------------------
# PyCBC comparison (optional)
# ----------------------------------------------------------------------------
SYNTH_MASS1, SYNTH_MASS2 = 1000, 26
SYNTH_DELTA_F, SYNTH_F_LOWER, SYNTH_F_FINAL = 0.25, 20, 1024
