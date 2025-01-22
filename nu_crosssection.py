import numpy as np
from numpy import pi, cos, sin, sqrt, power

import sys
sys.path.append("../")

from alplib.constants import *
from scipy.integrate import quad




SSW_DAT = np.genfromtxt("data/sw2_theory_curve_Q2.txt")
def sw2_running(q):
    return np.interp(q, SSW_DAT[:,0], SSW_DAT[:,1])




def dsigma_dEr_eves(Er, Enu, sw2, flavor="mu", gL_mod=1.0, gR_mod=1.0, running=False):
    """
    Takes in flavors "e", "mu", "tau", "ebar", "mubar", "taubar"
    """
    delta = "e" in flavor
    prefactor = 2 * G_F**2 * M_E / pi
    if running:
        q = 1e-3*sqrt(2*M_E*Er)
        sw2 = sw2_running(q)
    gL = delta + (sw2 - 0.5)*gL_mod
    gR = sw2*gR_mod
    if "bar" in flavor:
        return prefactor*((gR)**2 + power(gL * (1-Er/Enu), 2) - gL*gR*M_E*Er/power(Enu,2))

    return prefactor*((gL)**2 + power(gR * (1-Er/Enu), 2) - gL*gR*M_E*Er/power(Enu,2))




def dsigma_dEr_eves_running(Er, Enu, flavor="mu"):
    """
    Takes in flavors "e", "mu", "tau", "ebar", "mubar", "taubar"
    """
    delta = "e" in flavor
    prefactor = 2 * G_F**2 * M_E / pi

    q = 1e-3*sqrt(2*M_E*Enu)
    gL = delta + (sw2_running(q) - 0.5)
    gR = sw2_running(q)
    if "bar" in flavor:
        return prefactor*((gR)**2 + power(gL * (1-Er/Enu), 2) - gL*gR*M_E*Er/power(Enu,2))

    return prefactor*((gL)**2 + power(gR * (1-Er/Enu), 2) - gL*gR*M_E*Er/power(Enu,2))




def dsigma_dEr_eves_noME(Er, Enu, sw2, flavor="mu", gL_mod=1.0, gR_mod=1.0):
    """
    Takes in flavors "e", "mu", "tau", "ebar", "mubar", "taubar"
    """
    delta = "e" in flavor
    prefactor = 2 * G_F**2 * M_E / pi
    gL = delta + (sw2 - 0.5)*gL_mod
    gR = sw2*gR_mod
    if "bar" in flavor:
        gL = sw2*gR_mod
        gR = (sw2 - 0.5)*gL_mod + delta
    return prefactor*((gL)**2 + power(gR * (1-Er/Enu), 2))



def Enu_min(Er):
    return (sqrt(Er**2 + 2*M_E*Er) + Er)/2

def eves_total_xs(Enu, sw2, n_electrons=1):
    # returns total xs in cm^2
    return n_electrons * HBARC**2 * quad(dsigma_dEr_eves, 0.0, Enu/(1+M_E/(2*Enu)), args=(Enu, sw2,))[0]

# data that contains the cross section / Enu in GeV per 1e-38 cm^2
total_xs_dat = np.genfromtxt("data/total_nu_xs_perCM2_perEnu_GeV.txt")
def total_nu_xs(Enu, N_nucleons=12):
    # returns total XS in cm^2
    return N_nucleons * 1e-38 * (1e-3*Enu)*np.interp(1e-3*Enu, total_xs_dat[:,0], total_xs_dat[:,1])