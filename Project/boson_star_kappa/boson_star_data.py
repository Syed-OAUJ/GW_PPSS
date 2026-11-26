"""
boson_star_data.py -- loading the boson-star (mass, spin, kappa) table and
building fiducial binaries from it.
Data: tamaraevst/Spin-induced-quadrupole-moments-of-boson-stars
"""
import pandas as pd

import config as cfg
from waveform import masses_to_system


def _read(path=cfg.BOSON_STAR_FILE, usecols=None):
    return pd.read_csv(path, sep=r"\s+", comment="#",
                       names=cfg.BOSON_STAR_COLUMNS, usecols=usecols)


def load_kappa_chi_table(path=cfg.BOSON_STAR_FILE):
    """Full table plus chi = J/M^2 (no explicit spin column in the file)."""
    df = _read(path)
    df["chi"] = df["ang"] / df["mass"] ** 2
    return df.drop_duplicates(subset=["chi"]).sort_values("chi").reset_index(drop=True)


def kappa_from_chi(chi_query, table=None):
    """
    Nearest-neighbour kappa at a given chi. The sequence is NOT single-valued in
    chi (several branches share it), so this does not interpolate.
    """
    table = load_kappa_chi_table() if table is None else table
    row = table.loc[(table["chi"] - chi_query).abs().idxmin()]
    return float(row["kappa"]), float(row["chi"])


def load_mass_kappa_table(path=cfg.BOSON_STAR_FILE):
    return _read(path, usecols=["mass", "kappa"])


def fiducial_from_rows(row1=cfg.BOSON_STAR_ROWS[0], row2=cfg.BOSON_STAR_ROWS[1],
                       path=cfg.BOSON_STAR_FILE):
    """
    Fiducial binary from two table rows: body 1 <- row1, body 2 <- row2.
    Returns masses [Msun], kappas, kappa_s/kappa_a, and M, eta, Mc in seconds.
    """
    table = load_mass_kappa_table(path)
    r1, r2 = table.iloc[row1], table.iloc[row2]
    m1, k1 = float(r1["mass"]), float(r1["kappa"])
    m2, k2 = float(r2["mass"]), float(r2["kappa"])
    M, eta, Mc = masses_to_system(m1, m2)
    return dict(m1=m1, kappa1=k1, m2=m2, kappa2=k2,
                kappa_s=(k1 + k2) / 2, kappa_a=(k1 - k2) / 2,
                M=M, eta=eta, Mc=Mc)
