"""
MEE_Thermo_Func.py  --  black-liquor thermophysical property functions

Correlations: Gullichsen & Fogelholm, as tabulated in
Kumar, D., Kumar, V., Singh, V.P. (2013), "Modeling and dynamic simulation of
mixed feed multi-effect evaporators in paper industry", Applied Mathematical
Modelling 37, 384-397, Table 1.

UNIT CONVENTIONS (matching the MEE model):
    T   : temperature           [degC]
    Bx  : solids content        [%]      (correlations use fraction X = Bx/100)
    P   : pressure              [kPa abs]
    h   : specific enthalpy     [MJ/t]   (numerically equal to kJ/kg)
    Cp  : specific heat         [MJ/(t.degC)]  (= kJ/(kg.K))
    rho : density               [t/m3]
    dh  : latent heat           [MJ/t]

NOTE: kJ/kg == MJ/t and kJ/(kg.K) == MJ/(t.degC), so the Kumar/Gullichsen
values drop in directly without numeric conversion; only Bx% -> fraction and
density kg/m3 -> t/m3 are converted below.
"""

import numpy as np


# ------------------------------------------------------------------
# Internal helper: boiling-point rise (not imported by the main model)
# ------------------------------------------------------------------
def fBPR(T, Bx):
    """Boiling-point rise [degC]. T = liquor temperature [degC], Bx [%].
    (Liquor T used in the temperature-correction factor in place of vapour
    temperature to avoid an implicit solve; small approximation.)"""
    X = Bx / 100.0
    base = 6.173 * X - 7.48 * X**1.5 + 32.747 * X**2
    return base * (1.0 + 0.006 * (T - 3.7316))


# ------------------------------------------------------------------
# Water saturation pressure / temperature (Antoine form, unchanged)
# ------------------------------------------------------------------
def fP_eq(T):
    """Pure-water saturation pressure [kPa] at temperature T [degC]."""
    return 10 ** (7.8656 - 2188.8 / (T + 273.15))


def fT_eq(P):
    """Pure-water saturation temperature [degC] at pressure P [kPa]."""
    return 2188.8 / (7.8656 - np.log10(P)) - 273.15


def fT_eq_w(P):
    """Saturation temperature of pure water [degC] at pressure P [kPa]
    (condenser side; identical relation to fT_eq)."""
    return 2188.8 / (7.8656 - np.log10(P)) - 273.15


# ------------------------------------------------------------------
# Liquor equilibrium pressure (accounts for boiling-point rise)
# ------------------------------------------------------------------
def f_PJ_eq(T, Bx):
    """Equilibrium vapour pressure of black liquor [kPa] at liquor
    temperature T [degC] and solids Bx [%].  Equal to the pure-water
    saturation pressure evaluated at (T - BPR)."""
    return fP_eq(T - fBPR(T, Bx))


# ------------------------------------------------------------------
# Density, specific heat
# ------------------------------------------------------------------
def frho(T, Bx):
    """Black-liquor density [t/m3]. T [degC], Bx [%]."""
    X = Bx / 100.0
    rho_kg_m3 = (997.0 + 649.0 * X) * (
        1.008 - 0.237 * (T / 1000.0) - 1.94 * (T / 1000.0) ** 2
    )
    return rho_kg_m3 / 1000.0


def fcP(T, Bx):
    """Black-liquor specific heat [MJ/(t.degC)] (= kJ/(kg.K)). T [degC], Bx [%]."""
    X = Bx / 100.0
    return (
        4.216 * (1.0 - X)
        + (1.675 + 3.31 * T / 1000.0) * X
        + (4.87 - 20.0 * T / 1000.0) * (1.0 - X) * X**3
    )


# ------------------------------------------------------------------
# Enthalpies
# ------------------------------------------------------------------
def fh(T, Bx):
    """Black-liquor specific enthalpy [MJ/t], sensible heat referenced to 0 degC:
    h = Cp(T,Bx) * T.  Consistent with fcP as its temperature derivative."""
    return fcP(T, Bx) * T


def fh_ev(T, Bx):
    """Enthalpy of vapour evolved from the liquor [MJ/t]. Saturated steam
    enthalpy evaluated at the vapour temperature Tv = T - BPR. T [degC], Bx [%]."""
    Tv = T - fBPR(T, Bx)
    return 1.75228 * Tv + 2503.35


# ------------------------------------------------------------------
# Latent heat of vaporisation
# ------------------------------------------------------------------
def fdhvap(T):
    """Latent heat of vaporisation [MJ/t] at temperature T [degC]."""
    return 2519.5 - 2.653 * T


def fdhvap_w(T):
    """Latent heat of vaporisation of water [MJ/t] at temperature T [degC]
    (condenser side; identical relation to fdhvap)."""
    return 2519.5 - 2.653 * T