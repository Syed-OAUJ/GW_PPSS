import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
#from scipy.integrate import trapezoid
import pandas as pd


df = pd.read_csv(
    "./Numerical_PSD/aLIGO_ZeroDetHighPower_psd.txt",
    sep=r"\s+",       # handles any number of spaces/tabs
    header=None,      # no header row in the file
    names=["freq", "psd"]
)

# ---------------------------------------------------------------
# Symbols
# ---------------------------------------------------------------
f, f0, fs, S0, x = sp.symbols('f f0 fs S0 x', positive=True)

# ---------------------------------------------------------------
# New fit, Eq. (3.7a): f0 = 215 Hz, fs = 20 Hz, S0 = 1e-49 Hz^-1
#
#   Sh(f) = S0 [ x^-4.14 - 5 x^-2 + 111(1 - x^2 + x^4/2) / (1 + x^2/2) ],  f >= fs
#   Sh(f) = infinity,                                                     f <  fs
# ---------------------------------------------------------------
x_expr = f / f0

Sh_new_expr = S0 * (
        x ** sp.Rational(-414, 100)  # x^-4.14
        - 5 * x ** -2
        + 111 * (1 - x ** 2 + x ** 4 / 2) / (1 + x ** 2 / 2)
)

Sh_new_of_f = Sh_new_expr.subs(x, x_expr)

# Parameter values for the new fit
new_params = {f0: 215, fs: 20, S0: sp.Float(1e-49)}

# ---------------------------------------------------------------
# Older PSD used in Refs. [6,22,25,46]: f0 = 70 Hz, fs = 10 Hz, S0 = 6e-49 Hz^-1
#
#   Sh(f) = S0 [ x^-4 + 2 + 2 x^2 ],  f >= fs
#   Sh(f) = infinity,                 f <  fs
# ---------------------------------------------------------------
Sh_old_expr = S0 * (x ** -4 + 2 + 2 * x ** 2)
Sh_old_of_f = Sh_old_expr.subs(x, x_expr)

old_params = {f0: 70, fs: 10, S0: sp.Float(6e-49)}

# ---------------------------------------------------------------
# Substitute numeric parameters, then lambdify (as functions of f only)
# ---------------------------------------------------------------
Sh_new_numeric = Sh_new_of_f.subs(new_params)
Sh_old_numeric = Sh_old_of_f.subs(old_params)

Sh_new_func = sp.lambdify(f, Sh_new_numeric, 'numpy')
Sh_old_func = sp.lambdify(f, Sh_old_numeric, 'numpy')

# ---------------------------------------------------------------
# Evaluate on a log-spaced frequency grid, respecting each fs cutoff
# ---------------------------------------------------------------
freqs_new = np.logspace(np.log10(20), np.log10(2000), 500)  # fs = 20 Hz
freqs_old = np.logspace(np.log10(10), np.log10(2000), 500)  # fs = 10 Hz

#sqrtSh_new = np.sqrt(Sh_new_func(freqs_new))
#sqrtSh_old = np.sqrt(Sh_old_func(freqs_old))

# ---------------------------------------------------------------
# Plot: amplitude spectral density sqrt(Sh(f))  [Hz^-1/2]
# ---------------------------------------------------------------
plt.figure(figsize=(7, 6))
plt.loglog(freqs_new, Sh_new_func(freqs_new), 'k-', lw=2, label=r'Adv. LIGO (Eq. 3.7, $f_0=215$ Hz)')
plt.loglog(freqs_old, Sh_old_func(freqs_old), 'b--', lw=2, label=r'Adv. LIGO (older fit, $f_0=70$ Hz)')
plt.loglog(df['freq'], df['psd'], label='Data')

plt.xlabel('Frequency (Hz)')
plt.ylabel(r'$S_h(f)$  (Hz)')
plt.title('Comparison of Advanced LIGO noise PSD fits')
plt.legend()
plt.grid(True, which='both', ls=':', alpha=0.6)
plt.xlim(10, 2000)
plt.tight_layout()
plt.savefig('aligo_psd_comparison.png', dpi=150)
print("Saved plot.")

# Also print the symbolic expressions for reference
print("\nNew fit Sh(f):")
sp.pprint(Sh_new_of_f)
print("\nOlder fit Sh(f):")
sp.pprint(Sh_old_of_f)
