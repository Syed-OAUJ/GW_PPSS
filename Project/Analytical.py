r"""
SymPy translation of AISS_3p5NonSpinWaveform_AlsoKappa.nb
==========================================================

This module is a line-by-line translation of the Mathematica notebook into
SymPy. It is organised into the same three sections ("Titles") as the
original notebook:

    1. 3.5 PN Non-Spinning Waveform
    2. PSD: AdvLIGO and ET (Analytical)
    3. AISS Only Kappa  (spin-aligned phasing with the kappa_s / kappa_a
       quadrupole-monopole parameters, and the full waveform hAISSKappa)

Notes on the translation
-------------------------
* Mathematica's `Module[{...}, body]` local-variable blocks are translated
  into ordinary Python function bodies with local variables.
* Mathematica `/.` (ReplaceAll) substitutions are applied directly by
  writing the substituted expressions (delta, chiadl, chisdl, kappap,
  kappam) in terms of chi1, chi2, kappas, kappaa from the start, exactly as
  the notebook's replacement rules do.
* `\[Pi]fac` in the original notebook is a distinct symbol from `\[Pi]`
  everywhere it appears (in `v[f,M]`, inside `AISSKappa`, and inside
  `hAISSKappa`). Since it is never assigned a different value anywhere in
  the notebook and always plays the role of Pi, it is identified with
  `sympy.pi` here (`pifac = sympy.pi`). If your full notebook defines
  `\[Pi]fac` differently elsewhere, just change the `pifac` assignment
  below.
* `\[Gamma]E`, `\[Theta]`, `\[Lambda]` are the numeric constants set at the
  top of the first section (`gammaE = EulerGamma`, `theta = -1.28`,
  `lambda_ = -0.6451`). They are reused, unmodified, inside `AISSKappa`
  exactly as in the notebook (which relies on them being global there too).
* The first section's `psi[f_,mchirp_,tc_,phic_,eta_]` is transcribed
  faithfully -- note that in the original notebook this function only sums
  alpha0 through alpha3 (alpha4-alpha7 are computed above it but not
  actually used in `psi`). This looks like a leftover/incomplete
  definition in the notebook itself, but it is kept as-is here so the
  translation is faithful. The complete-through-3.5PN phasing is the one
  built in section 3 (`AISSKappa` / `hAISSKappa`).
"""
import sympy
import sympy as sp
from sympy import (
    symbols, Rational, sqrt, log, pi, exp, I, conjugate, simplify,
    lambdify, N, EulerGamma
)
import matplotlib.pyplot as plt
import numpy as np
from pycbc.catalog import Merger
from pycbc.filter import highpass_fir, lowpass_fir
from pycbc.psd import welch, interpolate
from pycbc.waveform import get_td_waveform, get_fd_waveform, td_approximants, fd_approximants


# ---------------------------------------------------------------------
# Section 1: "3.5 PN Non - Spinning Waveform"
# ---------------------------------------------------------------------

# Symbols (all assumed real, f/mchirp/eta positive as physical quantities)
f, mchirp, tc, phic, eta = symbols('f mchirp tc phic eta', real=True)
cuttoffFns = symbols('cuttoffFns', positive=True)

lambda_ = sp.Float(-0.6451)
theta_ = sp.Float(-1.28)
gammaE = N(EulerGamma)          # N[EulerGamma]

m_expr = mchirp / eta**Rational(3, 5)                       # m = mchirp/eta^(3/5)
v_expr = (pi * m_expr * f) ** Rational(1, 3)                 # v = (pi*m*f)^(1/3)
vlso_expr = (pi * m_expr * cuttoffFns) ** Rational(1, 3)     # vlso

alpha0 = sp.expand(Rational(3) / (128 * eta * v_expr**5))    # 3/(128 eta v^5) // Expand
alpha1 = alpha0 * 0
alpha2 = alpha0 * (Rational(20, 9) * (Rational(743, 336) + Rational(11, 4) * eta))
alpha3 = alpha0 * (-16 * pi)
alpha4 = alpha0 * 10 * (
    Rational(3058673, 1016064)
    + Rational(5429, 1008) * eta
    + Rational(617, 144) * eta**2
)
alpha5 = alpha0 * pi * (
    Rational(38645, 756)
    + Rational(38645, 252) * log(v_expr / vlso_expr)
    + Rational(5, 3) * eta * (1 + 3 * log(v_expr / vlso_expr))
)
alpha6 = alpha0 * (
    (Rational(11583231236531, 4694215680) - Rational(640, 3) * pi**2 - Rational(6848, 21) * gammaE)
    + eta * (
        Rational(-15335597827, 3048192)
        + Rational(2255, 12) * pi**2
        - Rational(1760, 3) * theta_
        + Rational(12320, 9) * lambda_
    )
    + Rational(76055, 1728) * eta**2
    - Rational(127825, 1296) * eta**3
    - Rational(6848, 21) * log(4 * v_expr)
)
alpha7 = alpha0 * pi * (
    Rational(77096675, 254016)
    + Rational(1014115, 3024) * eta
    - Rational(36868, 378) * eta**2
)

termtc = 2 * pi * f * tc
termphic = -phic


def psi(f_, mchirp_, tc_, phic_, eta_):
    """
    Direct translation of:
        psi[f_,mchirp_,tc_,phic_,eta_] :=
            termtc + termphic + alpha0 + alpha1*v + alpha2*v^2 + alpha3*v^3 - Pi/4

    As in the notebook, this expression is only correct when called with the
    same symbols used to build termtc/alpha0-3/v above (f, mchirp, tc, phic,
    eta). Only alpha0..alpha3 enter this particular psi (alpha4-alpha7 are
    computed in section 1 but not used here -- see module docstring).
    """
    return termtc + termphic + alpha0 + alpha1 * v_expr + alpha2 * v_expr**2 + alpha3 * v_expr**3 - pi / 4


def h(f_, mchirp_, tc_, phic_, eta_):
    """h[f_,mchirp_,tc_,phic_,eta_] := f^(-7/6) * Exp[I*psi[f,mchirp,tc,phic,eta]]"""
    return f_**Rational(-7, 6) * exp(I * psi(f_, mchirp_, tc_, phic_, eta_))


psi_f = psi(f, mchirp, tc, phic, eta)
h1 = h(f, mchirp, tc, phic, eta)

# conjh1 = Conjugate[h1] /. Conjugate[x_]:>(x /. a_Complex:>Conjugate[a])
# i.e. conjugate everything *except* treat the real symbols as real (only
# literal complex numbers get conjugated). With f, mchirp, tc, phic, eta
# declared real above, sympy's conjugate() reproduces exactly this behaviour.
conjh1 = conjugate(h1)


# ---------------------------------------------------------------------
# Section 2: "PSD : AdvLigo and ET (Analytical)"
# ---------------------------------------------------------------------

def PSDAjith(f_):
    """Advanced LIGO analytical PSD fit (Ajith)."""
    x = f_ / sp.Float(245.4)
    return sp.Float(10)**-48 * (
        sp.Float(0.0152) * x**-4
        + sp.Float(0.2935) * x**Rational(9, 4)
        + sp.Float(2.7951) * x**Rational(3, 2)
        - sp.Float(6.5080) * x**Rational(3, 4)
        + sp.Float(17.7622)
    )


def PSDET(f_):
    """Einstein Telescope analytical PSD fit."""
    x = f_ / sp.Float(100)
    return sp.Float(10)**-50 * (
        sp.Float(2.39) * sp.Float(10)**-27 * x**sp.Float(-15.64)
        + sp.Float(0.349) * x**sp.Float(-2.145)
        + sp.Float(1.76) * x**sp.Float(-0.12)
        + sp.Float(0.409) * x**sp.Float(1.10)
    )


# ---------------------------------------------------------------------
# Section 3: "AISS Only Kappa"
# ---------------------------------------------------------------------

# NOTE: \[Pi]fac is identified with pi -- see module docstring.
pifac = pi

M_sym, eta_s, chi1, chi2, kappas, kappaa = symbols(
    'M eta chi1 chi2 kappas kappaa', real=True
)


def v(f_, M_):
    """v[f_,M_] := (pifac*M*f)^(1/3)"""
    return (pifac * M_ * f_) ** Rational(1, 3)


def AISSKappa(f_, M_, eta_, chi1_, chi2_, kappas_, kappaa_):
    """
    Direct translation of:

        AISSKappa[f_,M_,eta_,chi1_,chi2_,kappas_,kappaa_] := Module[...]

    Returns the 3.5PN spin-aligned + quadrupole-monopole (kappa_s, kappa_a)
    phasing psi(f), i.e. (alpha0 + Total[alphaList] - pifac/4).
    """
    vval = v(f_, M_)
    f0 = 10
    vlso = (pi * M_ * f0) ** Rational(1, 3)

    alpha0_ = Rational(3) / (128 * eta_ * vval**5)

    alpha2_ = alpha0_ * (Rational(20, 9) * (Rational(743, 336) + Rational(11, 4) * eta_))

    # substitutions used throughout: delta, chiadl, chisdl, kappap, kappam
    delta = sqrt(1 - 4 * eta_)
    chiadl = (chi1_ - chi2_) / 2
    chisdl = (chi1_ + chi2_) / 2
    kappap = 2 * kappas_
    kappam = 2 * kappaa_

    alpha3_ = alpha0_ * (
        -16 * pi
        + Rational(113, 3) * delta * chiadl
        + (Rational(113, 3) - Rational(76, 3) * eta_) * chisdl
    )

    alpha4_ = alpha0_ * (
        Rational(15293365, 508032)
        + Rational(27145, 504) * eta_
        + Rational(3085, 72) * eta_**2
        + (-Rational(5, 8) + 100 * eta_ - 25 * kappap + 50 * eta_ * kappap) * chiadl**2
        + (-50 * kappam + 100 * eta_ * kappam) * chiadl * chisdl
        + (-Rational(5, 8) - Rational(195, 2) * eta_ - 25 * kappap + 50 * eta_ * kappap) * chisdl**2
        + delta * (
            -25 * kappam * chiadl**2
            + (-Rational(5, 4) - 50 * kappap) * chiadl * chisdl
            - 25 * kappam * chisdl**2
        )
    )

    alpha5_ = alpha0_ * pi * (
        Rational(38645, 756)
        + Rational(38645, 252) * log(vval / vlso)
        + Rational(5, 3) * eta_ * (1 + 3 * log(vval / vlso))
    )

    alpha6_ = alpha0_ * (
        (Rational(11583231236531, 4694215680) - Rational(640, 3) * pi**2 - Rational(6848, 21) * gammaE)
        + eta_ * (
            Rational(-15335597827, 3048192)
            + Rational(2255, 12) * pi**2
            - Rational(1760, 3) * theta_
            + Rational(12320, 9) * lambda_
        )
        + Rational(76055, 1728) * eta_**2
        - Rational(127825, 1296) * eta_**3
        - Rational(6848, 21) * log(4 * vval)
    )

    alpha7_ = alpha0_ * pi * (
        Rational(77096675, 254016)
        + Rational(1014115, 3024) * eta_
        - Rational(36868, 378) * eta_**2
    )

    alphaList = [
        0,
        alpha2_ * vval**2,
        alpha3_ * vval**3,
        alpha4_ * vval**4,
        alpha5_ * vval**5,
        alpha6_ * vval**6,
        alpha7_ * vval**7,
    ]

    return alpha0_ + sum(alphaList) - pifac / 4


def hAISSKappa(f_, mchirp_, eta_, tc_, phic_, chi1_, chi2_, kappas_, kappaa_):
    """
    Direct translation of:

        hAISSKappa[f_,mchirp_,eta_,tc_,phic_,chi1_,chi2_,kappas_,kappaa_] := Module[...]

    Full frequency-domain waveform f^(-7/6) * Exp[I*phase].
    """
    M_ = mchirp_ / eta_**Rational(3, 5)
    psiAISSKappa = AISSKappa(f_, M_, eta_, chi1_, chi2_, kappas_, kappaa_)
    # (psiEXTRA / EccKappaPhasing2 term is commented out in the original
    #  notebook and therefore not included here either)
    phase = psiAISSKappa + 2 * pifac * f_ * tc_ - phic_
    return f_ ** Rational(-7, 6) * exp(I * phase)


# ---------------------------------------------------------------------
# Quick sanity check / demonstration
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Section 1: psi and h ===")
    sp.pprint(psi_f)
    print()
    print("h1 =")
    sp.pprint(h1)

    print("\n=== Section 2: PSDs at f=100 Hz ===")
    print("PSDAjith(100) =", N(PSDAjith(100)))
    print("PSDET(100)    =", N(PSDET(100)))

    print("\n=== Section 3: AISSKappa / hAISSKappa numeric check ===")
    vals = {
        f: 50.0, mchirp: 1.2, eta: 0.24, tc: 0.0, phic: 0.0,
        chi1: 0.1, chi2: -0.05, kappas: 1.0, kappaa: 0.0,
    }
    phase_expr = AISSKappa(f, mchirp / eta**Rational(3, 5), eta, chi1, chi2, kappas, kappaa)
    print("phase(50 Hz) =", N(phase_expr.subs(vals)))

    hval = hAISSKappa(f, mchirp, eta, tc, phic, chi1, chi2, kappas, kappaa)
    print("hAISSKappa(50 Hz) =", N(hval.subs(vals)))

    print("#-------------")
    print(" My code starts here")

    # Build the symbolic expression once, then compile it to a fast numpy function
    f_sym = symbols('f', positive=True)
    PSDET_expr = PSDET(f_sym)
    PSDET_numpy = lambdify(f_sym, PSDET_expr, 'numpy')

    PSDAjith_expr = PSDAjith(f_sym)
    PSDAjith_numpy = lambdify(f_sym, PSDAjith_expr, 'numpy')

    # Frequency range: log-spaced is more sensible than linear for a loglog plot
    freqs = np.logspace(0, 4, 2000)  # 1 Hz to 10,000 Hz
    psd_et_vals = PSDET_numpy(freqs)
    psd_aijth_vals = PSDAjith_numpy(freqs)

    # synthetic waveform from pycbc
    hp_synth, hc_synth = get_fd_waveform(
    approximant="IMRPhenomD",
    mass1=1000, mass2=26,       # GW150914-like masses (100+100 was solar-mass BHs, not really "SMBH")
    delta_f=1.0 / 4,
    f_lower=20,
    f_final=1024,
)
    # For simplicity i assume hp_synth and hc_synth obey basic pitagorean theorem
    h_net_synth = abs(hp_synth)**2 + abs(hc_synth)**2
    freqs_synth = h_net_synth.sample_frequencies.numpy()



    # --- measured GW150914 noise PSD (Welch estimate), one curve per IFO --
    measured_psds = {}
    for ifo in ['H1', 'L1']:
        strain = Merger("GW150914").strain(ifo)
        strain = highpass_fir(strain, 15, 8)
        psd = interpolate(welch(strain), 1.0 / strain.duration)
        measured_psds[ifo] = psd

    # --- single combined figure --------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(freqs, psd_et_vals, label="PSDET (analytical)")
    ax.loglog(freqs, psd_aijth_vals, label="PSDAjith (analytical)")
    ax.loglog(freqs_synth, h_net_synth, label="Synthetic Heavy BH")

    for ifo, psd in measured_psds.items():
        # restrict to the same frequency window for a fair comparison
        mask = (psd.sample_frequencies.numpy() >= 1) & (psd.sample_frequencies.numpy() <= 1e4)
        ax.loglog(psd.sample_frequencies.numpy()[mask], psd.numpy()[mask],
                  label=f"{ifo} measured PSD (GW150914)", alpha=0.8)

    ax.set_xlabel("Frequency f [Hz]")
    ax.set_ylabel(r"PSD$(f)$  [strain$^2$/Hz]")
    ax.set_title("Analytical vs. measured detector PSDs (GW150914)")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig("PSD_plot.png", dpi=150)
    print("saved PSD_plot.png")
