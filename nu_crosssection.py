import numpy as np
from numpy import pi, cos, sin, sqrt, power

import sys
sys.path.append("../")

from alplib.constants import *
from alplib.cross_section_mc import *

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


def dsigma_dEl_inv_lepton_decay(El, Enu, ml=M_MU):
    """
    Differential cross section for the inverse lepton decay.
    """
    return 2*G_F**2 * M_E * ((1 - El/Enu)**2 - (ml**2 - M_E**2)*(1 - El/Enu)/(2*M_E*Enu)) / pi



M_AXIAL = 1014.0

class NeutrinoNucleonCCQE:
    """
    Implements the CCQE cross section from https://arxiv.org/abs/1305.7513 (Formaggio, Zeller )
    """
    def __init__(self, flavor):
        self.flavor = flavor
        self.m_n = M_N
        self.m_lepton = M_E

        self.xi = 4.706  # Difference between proton and neutron magnetic moments.
        self.sign = -1

        if flavor == "mu" or flavor == "mubar":
            self.m_lepton = M_MU
        if flavor == "tau" or flavor == "taubar":
            self.m_lepton = M_TAU

        if flavor == "ebar" or flavor == "mubar" or flavor == "taubar":
            self.sign = 1
            self.m_n = M_P
    
    def p1_cm(self, s):
        return np.sqrt((np.power(s - self.m_n**2, 2))/(4*s))

    def p3_cm(self, s):
        return np.sqrt((np.power(s - self.m_lepton**2 - self.m_n**2, 2) - np.power(2*self.m_lepton*self.m_n, 2))/(4*s))

    def dsigma(self, qsq, ev):
        tau = qsq / (4 * self.m_n ** 2)
        GD = (1 / (1 + qsq / 710000) ** 2)  # Dipole form factor with vector mass.
        TE = np.sqrt(1 + (6e-6 * qsq) * np.exp(-qsq / 350000))  # Transverse Enhancement of the magnetic dipole.

        FA = -1.267 / (1 + (qsq / M_AXIAL)) ** 2  # Axial form factor.
        Fp = (2 * FA * (self.m_n) ** 2) / (M_PI0 ** 2 + qsq)  # Pion dipole form factor (only relevant for low ev).
        F1 = GD * ((1 + self.xi * tau * TE) / (1 + tau))  # First nuclear form factor in dipole approximation.
        F2 = GD * (self.xi * TE - 1) / (1 + tau)  # Second nuclear form factor in dipole approximation.

        # A, B, and C are the vector, pseudoscalar, and axial vector terms, respectively.
        A = ((self.m_lepton ** 2 + qsq) / self.m_n ** 2) * (
                (1 + tau) * FA ** 2 - (1 - tau) * F1 ** 2 + tau * (1 - tau) * (F2) ** 2 + 4 * tau * F1 * F2
                - 0.25 * ((self.m_lepton / self.m_n) ** 2) * ((F1 + F2) ** 2 + (FA + 2 * Fp) ** 2
                                                        - 4 * (tau + 1) * Fp ** 2))
        B = 4 * tau * (F1 + F2) * FA
        C = 0.25 * (FA ** 2 + F1 ** 2 + tau * (F2) ** 2)

        return ((1 / (8 * np.pi)) * (G_F * CABIBBO * self.m_n / ev) ** 2) * \
            (A + self.sign * B * ((4 * self.m_n * ev - qsq - self.m_lepton ** 2) / (self.m_n) ** 2)
                + C * ((4 * self.m_n * ev - qsq - self.m_lepton ** 2) / (self.m_n) ** 2) ** 2)

    def total_xs(self, ev):
        """
        Total cross section in cm^2
        """
        if ev < self.m_lepton + (self.m_lepton**2 / (2*self.m_n)):
            return 0.0

        sqts = np.sqrt(self.m_n ** 2 + 2 * self.m_n * ev)
        E_l = (sqts ** 2 + self.m_lepton ** 2 - self.m_n ** 2) / (2 * sqts)

        if E_l ** 2 < self.m_lepton ** 2:
            return 0.0
        
        q2_low = -self.m_lepton ** 2 + (sqts ** 2 - self.m_n ** 2) / (sqts) * \
                (E_l - np.sqrt(E_l ** 2 - self.m_lepton ** 2))
        q2_high = -self.m_lepton ** 2 + (sqts ** 2 - self.m_n ** 2) / (sqts) * \
                (E_l + np.sqrt(E_l ** 2 - self.m_lepton ** 2))

        return quad(self.dsigma, q2_low, q2_high, args=ev)[0]

    def rates(self, ev, n_samples=1000):
        # CoM Frame
        sqts = np.sqrt(self.m_n ** 2 + 2 * self.m_n * ev)
        E_l = (sqts ** 2 + self.m_lepton ** 2 - self.m_n ** 2) / (2 * sqts)

        if E_l ** 2 < self.m_lepton ** 2:
            return np.array([0]), np.array([0]), np.array([0])
        
        p_l = np.sqrt(E_l**2 - self.m_lepton**2)
        
        # Draw random values on the 2-sphere in CoM frame
        phi_rnd = 2*pi*np.random.ranf(n_samples)
        cos_theta_rnd = (1 - 2*np.random.ranf(n_samples))
        theta_rnd = np.arccos(cos_theta_rnd)

        q2_vals = -self.m_lepton ** 2 + (sqts ** 2 - self.m_n ** 2) / (sqts) * \
                (E_l + cos_theta_rnd*np.sqrt(E_l ** 2 - self.m_lepton ** 2))
        
        p1_cm = self.p1_cm(self.m_n**2 + 2*ev*self.m_n)
        p3_cm = self.p3_cm(self.m_n**2 + 2*ev*self.m_n)

        mc_volume = 4*p1_cm*p3_cm/n_samples
        dsigma_weights = self.dsigma(q2_vals, ev) * mc_volume

        # Boost to lab frame
        p_in_cm = LorentzVector(self.m_n, 0.0, 0.0, 0.0) + LorentzVector(ev, 0.0, 0.0, ev)
        v_in = Vector3(p_in_cm.p1 / p_in_cm.energy(),
                       p_in_cm.p2 / p_in_cm.energy(),
                       p_in_cm.p3 / p_in_cm.energy())

        p4_lepton_cm = [LorentzVector(E_l, p_l*np.cos(phi_rnd[i])*np.sin(theta_rnd[i]),
                                    p_l*np.sin(phi_rnd[i])*np.sin(theta_rnd[i]),
                                    p_l*cos_theta_rnd[i]) for i in range(n_samples)]
        p4_lepton_lab = [lorentz_boost(p4_lep, -v_in) for p4_lep in p4_lepton_cm]

        lepton_lab_energies = np.array([p4.energy() for p4 in p4_lepton_lab])
        lepton_lab_thetas = np.array([p4.theta() for p4 in p4_lepton_lab])

        return lepton_lab_energies, lepton_lab_thetas, dsigma_weights


