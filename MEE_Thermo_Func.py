import numpy as np


def fP_eq(T):
    # Saturation pressure (pure water)
    return 10 ** (7.8656 - 2188.8 / (T + 273.15))


def fT_eq(P):
    # Saturation temperature (pure water)
    return 2188.8 / (7.8656 - np.log10(P)) - 273.15


def f_PJ_eq(T, Bx):
    # Saturation pressure (juice)
    BPE = 6.054e-5 * (((T + 273.15) ** 2 * Bx ** 2) / ((374.3 - T) ** 2) * 5.84e-7 * (Bx - 40) ** 2 + 7.2e-4)
    Tvap = T - BPE
    return fP_eq(Tvap)


def fh(T, Bx):
    # Specific enthalpy (juice)
    return 2.326 * ((Bx / 10) * (100 + Bx) / (900 - 8 * Bx)
                    + 1.8 * T * (1 - Bx / 100) * (0.6 - 9e-4 * T))


def fdhvap(T):
    # Heat of vaporisation (pure water)
    Tr = (T + 273.15) / 647.13
    return 2889 * (1 - Tr) ** (0.3199 - 0.212 * Tr + 0.25795 * Tr ** 2)


def fh_ev(T, Bx):
    # Specific enthalpy (water vapour)
    return fh(T, Bx) + fdhvap(T)


def fcP(T, Bx):
    # Specific heat capacity (juice)
    return (4.1253 - 0.02804 * Bx + 6.7e-5 * Bx * T
            + 1.8691e-3 * T - 9.271e-6 * T ** 2)


def frho(T, Bx):
    # Density (juice)
    return (1 + Bx * (Bx + 200) / 54e3) * (1 - 0.036 * (T - 20) / (160 - T))


def fT_eq_w(P):
    # Saturation temperature of pure water [°C] from pressure P [kPa]
    return 2188.8 / (7.8656 - np.log10(P)) - 273.15


def fdhvap_w(T):
    # Heat of vaporisation of pure water [MJ/t] at temperature T [°C]
    Tr = (T + 273.15) / 647.13

    return 2889 * (1 - Tr) ** (0.3199 - 0.212 * Tr + 0.25795 * Tr**2)