import numpy as np
from numpy import pi, cos, sin, sqrt, power
import mpmath as mp
from scipy.integrate import quad

import sys
sys.path.append("../")

from alplib.constants import *
from alplib.cross_section_mc import *




SSW_DAT = np.genfromtxt("data/sw2_theory_curve_Q2.txt")
def sw2_running(q):
    # Takes q in GeV
    return np.interp(q, SSW_DAT[:,0], SSW_DAT[:,1])




def dsigma_dEr_eves(Er, Enu, sw2, flavor="mu", gL_mod=1.0, gR_mod=1.0, cr=0.0, running=False):
    """
    Takes in flavors "e", "mu", "tau", "ebar", "mubar", "taubar"
    takes the charge radius (cr) in units of cm^2
    """
    delta = "e" in flavor
    prefactor = 2 * G_F**2 * M_E / pi
    cr_prefactor = (M_W**2 * sw2 / 3) / HBARC**2
    if running:
        q = 1e-3*sqrt(2*M_E*Er)
        sw2 = sw2_running(q)
    gL = delta + (sw2 - 0.5)*gL_mod + cr_prefactor*cr
    gR = sw2*gR_mod + cr_prefactor*cr
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




# Tau neutrino total CC cross section
nutau_ccxs_dat = np.genfromtxt("data/xs/nutau_totalCCXS_perGeV_per1e-38cm2.txt")
def nutau_cc_xs(Enu):
    # takes Enu in MeV
    # returns XS in cm^2
    EnuGeV = 1e-3 * Enu
    return 1e-38 * EnuGeV * np.interp(EnuGeV, nutau_ccxs_dat[:,0], nutau_ccxs_dat[:,1], left=0.0)




class EvESCrossSectionNLO:
    """
    EvES cross section @ NLO [Tomalak and Hill, 1907.03379]
    """
    def __init__(self, species='e', sw2=SSW, nlo=True):
        self.cp = 1
        if 'bar' in species:
            self.cp = -1
        self.species = 'e'
        if 'mu' in species:
            self.species = 'mu'
        if 'tau' in species:
            self.species = 'tau'
        
        self.nlo = nlo
        self.sw2 = sw2
        
        self.delta = ("e" in self.species)
        self.cc_prefactor = 2 * np.sqrt(2) * G_F
        self.xs_prefactor = M_E / (4 * np.pi)

        self.cL_nul = self.cc_prefactor * (sw2 - 0.5 + self.delta)
        self.cR = self.cc_prefactor * sw2

        self.cL_l = self.cc_prefactor * (-0.5 + sw2)
        self.cL_u = self.cc_prefactor * (0.5 - (2/3)*sw2)
        self.cL_d = self.cc_prefactor * (-0.5 + (1/3)*sw2)
        self.cR_l = self.cc_prefactor * sw2
        self.cR_u = -self.cc_prefactor * (2/3) * sw2
        self.cR_d = -self.cc_prefactor * (-1/3) * sw2
    
    # Momentum dependent couplings
    def cL_nul_running(self, El):
        q = 1e-3*np.sqrt(2 * M_E * (El - M_E))  # must pass in GeV
        return self.cc_prefactor * (sw2_running(q) - 0.5 + self.delta)
    
    def cR_running(self, El):
        q = 1e-3*np.sqrt(2 * M_E * (El - M_E))  # must pass in GeV
        return self.cc_prefactor * sw2_running(q)
    
    # Kinematical factors
    def IR(self, El, Enu):
        EnuPrime = M_E + Enu - El
        return np.power(EnuPrime / Enu, 2)
    
    def ILR(self, El, Enu):
        EnuPrime = M_E + Enu - El
        return - (M_E / Enu) * (1 - EnuPrime/Enu)

    # QCD and vertex corrections:
    def PiFF(self, q2, mf, mu):

        leading_log = (1/3) * np.log((mu**2) / (mf**2))
        
        x = 2.0 * (mf**2) / q2
        
        adim_piece = (5/9) - (2*x/3) + (1/3) * (1 - x) * np.sqrt(1 + 2*x) \
                        * np.log( (np.sqrt(1 + 2*x) - 1)/(np.sqrt(1 + 2*x) + 1) )
        
        polynomial_large_x = -(2/15) / x + (1/35) / (x**2)
        
        # To avoid machine precision errors when evaluating -inf + inf, use polynomial approx after x>100

        return leading_log + np.heaviside(100.0 - x, 1.0) * adim_piece \
                            + np.heaviside(x - 100.0, 0.0) * polynomial_large_x

    # Soft corrections
    def delta_soft(self, beta, lam=1e-7, eps=1e-2):
        # eps: the cutoff energy for the soft photon << M_E

        z = (1 - beta) / (1 + beta)
        L = np.log((1 + beta) / (1 - beta))

        plyl = np.vectorize(lambda x: (mp.polylog(2, x)).real)
        plyl_of_z = (plyl(z)).astype(float)

        term1 = (plyl_of_z - (np.pi**2) / 6) / beta
        term2 = -(2 / beta) * (beta - 0.5 * L) * np.log((2 * eps) / lam)
        term3 = (L / (2 * beta)) * (1 + np.log(np.sqrt(1 - beta**2) * (1 + beta) / (4 * beta**2)))

        return term1 + term2 + term3 + 1

    def deltaI(self, beta, omega, omegap, *,  eps=1e-2, tiny=1e-15):
        rho = np.sqrt(np.clip(1.0 - beta**2, 0.0, None))

        denom = (2.0 * beta / np.where(rho == 0.0, np.inf, rho)) * M_E * omegap
        cosDelta = (omega**2 - (beta**2 * M_E**2) / np.where(rho == 0.0, np.inf, rho**2) - omegap**2) / denom
        cosDelta = np.clip(cosDelta, -1.0, 1.0)

        Lbeta = np.log((1.0 + beta) / (1.0 - beta))

        # Argument of the last log; protect against cosΔ -> -1 and any tiny/zero denominators
        denom_log = beta * M_E * (1.0 + cosDelta)
        arg = (2.0 * (1.0 + beta) * eps) / np.where(denom_log > 0.0, denom_log, tiny)
        arg = np.where(arg > 0.0, arg, tiny)

        return (2.0 / beta) * (beta - 0.5 * Lbeta) * np.log(arg)

    def deltaII(self, beta, omega, omegap, eps=1e-15):
        rho = np.sqrt(np.clip(1.0 - beta**2, 0.0, None))
        # cosΔ formula
        denom = (2.0 * beta / np.where(rho == 0.0, np.inf, rho)) * M_E * omegap
        cosDelta = (omega**2 - (beta**2 * M_E**2) / np.where(rho == 0.0, np.inf, rho**2) - omegap**2) / denom

        # Numerically clamp cosΔ to [-1,1] to avoid slight excursions
        cosDelta = np.clip(cosDelta, -1.0, 1.0)

        # Building blocks
        Lbeta = np.log((1.0 - beta) / (1.0 + beta))  # < 0 for 0<β<1

        # Safe arguments for logs
        A = (rho * (cosDelta + 1.0)) / (4.0 * beta)
        A = np.where(A > 0.0, A, eps)  # avoid log(0) or negative due to rounding
        B = (1.0 - beta * cosDelta) / np.where(rho > 0.0, rho, np.inf)
        B = np.where(B > 0.0, B, eps)

        # Dilogarithm arguments
        z1 = (1.0 - beta) / (1.0 + beta)                              # in (0,1)
        z2 = (cosDelta - 1.0) / (cosDelta + 1.0 + 0.0)                # ≤ 0
        z3 = z2 * (1.0 + beta) / (1.0 - beta)                         # typically ≤ 0

        # Assemble terms
        plyl = np.vectorize(lambda x: (mp.polylog(2, x)).real)

        plyl1 = (plyl(z1)).astype(float)
        plyl2 = (plyl(z2)).astype(float)
        plyl3 = (plyl(z3)).astype(float)

        t_poly = (0.5 + np.log(A)) * Lbeta \
                - plyl1 -  plyl2 +  plyl3 \
                + (np.pi**2)/6.0

        out = (t_poly / beta) + np.log(B) - 1.0
        return out

    def deltav(self, beta, lam=1e-7):
        # domain: 0 < beta < 1
        if np.any(beta <= 0) or np.any(beta >= 1):
            raise ValueError("beta must be in the range (0, 1) for deltav calculation.")
        rho  = np.sqrt(1 - beta**2)
        Lb   = np.log((1 + beta) / (1 - beta))

        termA = (beta - 0.5 * Lb) * np.log(M_E / lam)
        termB = ((3 + rho) / 8) * Lb
        termC = -(1/8) * Lb * np.log(2 * (1 + rho) / rho)

        arg1 = (beta - 1 + rho) / (2 * beta)
        arg2 = (beta + 1 - rho) / (2 * beta)

        plyl = np.vectorize(lambda x: (mp.polylog(2, x)).real)

        plyl_arg1 = (plyl(arg1)).astype(float)
        plyl_arg2 = (plyl(arg2)).astype(float)

        termD = -0.5 * (plyl_arg1 - plyl_arg2)
        f1 = (termA + termB + termC + termD) / beta - 1

        return 2 * f1
    
    def dsigma_v(self, El, Enu, mu=2.0):
        beta = np.sqrt(1 - np.power(M_E/El, 2))
        f2 = (M_E / El) / (4 * beta) * np.log((1.0 - beta) / (1.0 + beta))

        dsigma_v = (ALPHA / np.pi) * self.dsigma_dEl_LO(El, Enu)

        return 0.  # TODO(AT): remove as its obsolete
    
    def dsigma_NF(self, El, Enu, mu=1.0):
        beta = np.sqrt(1 - np.power(M_E/El, 2))
        EnuPrime = M_E + Enu - El
        f2 = (M_E / El) / (4 * beta) * np.log((1.0 - beta) / (1.0 + beta))
        IL_nf = 1 + (ALPHA/np.pi) * f2 * (0.5 * self.ILR(El, Enu) - (EnuPrime/Enu))
        IR_nf = IL_nf
        ILR_nf = 2*(1 + self.IR(El, Enu) - EnuPrime/Enu) - self.ILR(El, Enu)
        dsigma = 0.0
        if self.cp == 1:
            dsigma = self.xs_prefactor * (self.cL_nul**2 * IL_nf + self.cR**2 * IR_nf \
                                          + self.cL_nul*self.cR*ILR_nf)
        elif self.cp == -1:
            dsigma = self.xs_prefactor * (self.cL_nul**2 * IR_nf + self.cR**2 * ILR_nf \
                                          + self.cL_nul*self.cR*ILR_nf)
        
        return dsigma

    def dsigma_dyn(self, El, Enu, mu):
        q2 = 2 * M_E * (El - M_E)

        # sum over leptons and heavy quarks (charm, bottom, top) in the loop
        cL_nulSq_dyn_lep = self.cL_nul * (self.cL_l + self.cR_l)
        cR_Sq_dyn_lep = self.cR * (self.cL_l + self.cR_l)
        cLcR_dyn_lep = 0.5 * (self.cL_nul + self.cR) * (self.cL_l + self.cR_l)

        cL_nulSq_dyn_up = self.cL_nul * (self.cL_u + self.cR_u)
        cR_Sq_dyn_up = self.cR * (self.cL_u + self.cR_u)
        cLcR_dyn_up = 0.5 * (self.cL_nul + self.cR) * (self.cL_u + self.cR_u)

        cL_nulSq_dyn_down = self.cL_nul * (self.cL_d + self.cR_d)
        cR_Sq_dyn_down = self.cR * (self.cL_d + self.cR_d)
        cLcR_dyn_down = 0.5 * (self.cL_nul + self.cR) * (self.cL_d + self.cR_d)

        dsigma_lep = self.dsigma_dEl_LO(El, Enu, cL_nulSq_dyn_lep, cR_Sq_dyn_lep, cLcR_dyn_lep)
        dsigma_up = self.dsigma_dEl_LO(El, Enu, cL_nulSq_dyn_up, cR_Sq_dyn_up, cLcR_dyn_up)
        dsigma_down = self.dsigma_dEl_LO(El, Enu, cL_nulSq_dyn_down, cR_Sq_dyn_down, cLcR_dyn_down)

        dsigma_dyn_lep_heaavy_quark = (ALPHA/np.pi) * (  -1. * self.PiFF(q2, M_E, mu) * dsigma_lep \
                                                        -1. * self.PiFF(q2, M_MU, mu) * dsigma_lep \
                                                        -1. * self.PiFF(q2, M_TAU, mu) * dsigma_lep \
                                                        + (2/3) * self.PiFF(q2, M_C, mu) * dsigma_up \
                                                        + (2/3) * self.PiFF(q2, M_TOP, mu) * dsigma_up \
                                                        + (-1/3) * self.PiFF(q2, M_BOTTOM, mu) * dsigma_down \
                                                    )
        dsigma_dyn_uds_reduced = self.dsigma_dEl_LO(El, Enu, self.cc_prefactor * self.cL_nul,
                                                    self.cc_prefactor * self.cR,
                                                    self.cc_prefactor * (self.cL_nul + self.cR))
        PiGammaGamma_2GeV = 3.597
        return dsigma_dyn_lep_heaavy_quark + (ALPHA/np.pi) * (PiGammaGamma_2GeV - 2*self.sw2 * PiGammaGamma_2GeV)*dsigma_dyn_uds_reduced

    def I_nonfact(self, xyzrqv, El, Enu):
        # Generic non-factorizable kinematic piece
        # takes in tuple of xi, yi, zi, ri, qi, vi (xyzqrv)
        xi, yi, zi, ri, qi, vi = xyzrqv
        prefactor = np.pi**2 / Enu**3

        beta = np.sqrt(1 - np.power(M_E/El, 2))
        rho = np.sqrt(1.0 - beta**2)
        l0 = M_E + Enu - El

        beta_doppler = (1+beta)/(1-beta)
        beta_plus_one_over_rho = (1 + beta)/rho

        log1_arg = (2*Enu/M_E) / (-1 + (1 + 2*Enu/M_E)/beta_plus_one_over_rho)
        log2_arg = (2*l0/M_E) / (1 + 2*Enu/M_E - beta_plus_one_over_rho)
        log3_arg = (1 - beta_plus_one_over_rho) / (beta_doppler - beta_plus_one_over_rho * (1 + 2*Enu/M_E))

        plyl = np.vectorize(lambda x: (mp.polylog(2, x)).real)

        plyl1 = (plyl(beta_plus_one_over_rho)).astype(float)
        plyl2 = (plyl(1 + 2*Enu/M_E)).astype(float)
        plyl3 = (plyl((1+2*Enu/M_E)/beta_plus_one_over_rho)).astype(float)

        print("zi = {}, yi = {}, log1 = {}".format(zi, yi, np.log(log1_arg)))
        print("xi = {}, ri = {}, log2 = {}, log3 = {}".format(xi, ri, np.log(log2_arg), np.log(log3_arg)))
        print("vi = {}, ply1 = {}, ply2 = {}, ply3 = {}".format(vi, plyl1, plyl2, plyl3))
        print("qi = {}, logbeta = {}".format(qi, np.log(beta_doppler)))

        return prefactor * (zi + yi*np.log(log1_arg) + xi*np.log(log2_arg) + ri*log(log3_arg) \
                            + vi*(plyl1 - plyl2 + plyl3 - np.pi**2 / 6) \
                            + qi*np.log(beta_doppler))

    def IL_nonfact(self, El, Enu):
        # Non-facttorizable kinematic pieces        
        beta = np.sqrt(1 - np.power(M_E/El, 2))
        rho = np.sqrt(np.clip(1.0 - beta**2, 0.0, None))

        vL = 0.5 * (M_E**2 / 2 + 2*M_E*Enu + Enu**2)
        xL = (-2/15 * Enu**5/M_E**3 + 1/3 * Enu**3/M_E
            + ( (1 + 3*beta**2)/(3*rho**3) - (4*beta**4 - 11*beta**2 + 7)/(3*rho**4) ) * Enu**2
            + (2/rho**3 - (beta**4 - beta**2 + 2)/rho**4) * M_E*Enu
            + ( (-7*beta**4 + 14*beta**2 - 22)/(15*rho**4) + (15*beta**4 - 25*beta**2 + 22)/(15*rho**5) ) * M_E**2 )
        yL = 0.5 * Enu * (Enu - M_E)
        rL = ( (-(2 + beta)/3 * rho/(1 + beta)**2 + (14 + beta)/6/(1 + beta))*Enu**2
            + ( (beta - rho**2)/(rho*(1 + beta)) + 0.5*(1 + 1/(1 + beta)**2) )*M_E*Enu
            + ( (-(17*beta**2 + 36*beta + 22))/(30*(1 + beta)**3)
                + (14*beta**2 + 43*beta + 44)/(60*(1 + beta)**2) )*M_E**2 )
        qL = ((1/2* rho/(1 + beta) - (1 + beta)/(2*beta))*Enu**2
            + beta/(2*rho)*M_E*Enu
            + (1/2* rho/(1 + beta) - (1 + beta)/(2*beta))*M_E**2)
        
        zL0 = (25*beta**2 - 49)/(60*rho**3) * (1 - 1/rho) - 8*beta**2/(15*rho**2)
        zLw = (-20*beta**3 + 51*beta**2 + 38*beta - 105)/(60*rho**3) - (55*beta**3 + 54*beta**2 - 82*beta - 105)/(60*rho**2*(1 + beta))
        zLw2 = (7*beta**2 + 8*beta - 23)/(30*(1 + beta)*rho) - (15*beta**2 + 6*beta - 23)/(30*rho**2)
        zLw3 = (3 - beta)/(30*rho) - (3 + 2*beta)/(30*(1+beta))
        zLw4 = 1/15 - rho/(15*(1 + beta))

        zL = (zLw4*Enu**4 + zLw3*M_E*Enu**3 + zLw2*Enu**2 * M_E**2 \
              + zLw*Enu*M_E**3 + zL0*M_E**4) / (M_E**2)
        
        return self.I_nonfact(np.array([xL, yL, zL, rL, qL, vL]), El=El, Enu=Enu)

    def IR_nonfact(self, El, Enu):
        beta = np.sqrt(1 - np.power(M_E/El, 2))
        rho = np.sqrt(np.clip(1.0 - beta**2, 0.0, None))
        l0 = M_E + Enu - El

        vR = 0.5 * (l0**2 + M_E**2 * (beta**2 + rho)/rho**2)
        xR = -l0**2 * (35*l0*M_E**2 - 10*l0**2 * M_E + 2*l0**3 - 30*M_E**3) / (15*M_E**3)
        yR = (-Enu**4 
            - 2*(5 - 1/rho)*M_E*Enu**3
            + (128*beta**2 + 11*rho - 16)/rho**2 * M_E**2*Enu**2
            + (6*beta**2 + 9*rho - 10)/rho**2 * M_E**3*Enu
            + (beta**2 + 2*rho - 2)/rho**2 * M_E**4) / (M_E + 2*Enu)**2
        qR = ((1/2* rho/(1 + beta) - (1 + beta)/(2*beta))*l0**2
            + ((2 - 1/(1 + beta)) - (2 - beta)/(2*rho))*M_E*l0
            + ((4*beta**3 + beta**2 - 4*beta + 2)/(4*beta*rho**2) 
                + (-beta**3 + 2*beta**2 + beta - 1)/(2*beta*rho*(1 + beta)))*M_E**2)
        rR = ( (-(2 + beta)/3 * rho/(1 + beta)**2 + (14 + beta)/6/(1 + beta))*l0**2
            + ( (beta**2 - 5*beta + 1)/(3*rho*(1 + beta)) 
                + (7*beta**2 + 8*beta - 2)/(6*(1 + beta)**2) )*M_E*l0
            + ( (-23*beta**3 + 14*beta**2 + 41*beta - 2)/(30*rho*(1 + beta)**2) 
                + (-28*beta*rho**2 + 43*beta**2 + 2)/(30*rho*(1 + beta)**2) )*M_E**2 )
        
        
        zw4 = (1/15) - (1/15) * rho / (1+beta)
        zRw4 = -8/(15*rho) + (18 - beta)/(15*(1 + beta))
        zRw3 = (113*beta**2 - 2*beta - 133)/(30*(1 + beta)*rho) - (143*beta**2 - 34*beta - 133)/(30*rho**2)
        zRw2 = (-339*beta**3 - 805*beta**2 - 353*beta + 851)/(60*rho**3) + (-760*beta**3 - 825*beta**2 + 778*beta + 851)/(60*rho**2*(1 + beta))
        zRw = (beta*((433 - 45*beta)*beta + 44) - 439)/(30*rho**3) + beta*(beta*(27*beta*(11*beta + 1) - 730) - 29) + 439/(30*rho**4)
        zR0 = (270*beta**2 - 269)/(60*rho**3) + (309*beta**4 - 839*beta**2 + 538)/(120*rho**4)
        zR = (2*zw4*Enu**5 + zRw4*M_E*Enu**4 + zRw3*M_E**2 * Enu**3 + zRw2*M_E**3 * Enu**2 \
            + zRw*Enu*M_E**4 + zR0*M_E**5) / (M_E**2*(M_E + 2*Enu))
        
        return self.I_nonfact(np.array([xR, yR, zR, rR, qR, vR]), El=El, Enu=Enu)

    def ILR_nonfact(self, El, Enu):
        beta = np.sqrt(1 - np.power(M_E/El, 2))
        rho = np.sqrt(np.clip(1.0 - beta**2, 0.0, None))
        l0 = M_E + Enu - El

        vLR = 0.5 * M_E * (2*l0 - M_E)
        xLR = (3*l0*M_E**2 - 3*l0**2*M_E - 2*l0**3 + 3*M_E**2*Enu) / (3*M_E)
        yLR = M_E * El * (1 - ((M_E + 2*Enu)**2 - M_E*Enu)/(El*(M_E + 2*Enu)))
        rLR = (1/3 * (7 + 5*beta/2 + (2*beta**2 - 4*beta - 7)/rho) * M_E**2/(1 + beta)
            + (1 + (1 - 2*rho)/(1 + beta))*M_E*Enu)
        qLR = ((1 - beta)/beta * Enu**2 
            - 2*M_E*Enu 
            + (1 + beta/2)/M_E * M_E**2 * (l0 - Enu/M_E) 
            + beta*M_E*El)
        zLR = (2*l0 + 9*M_E)/6 * (l0 - rho*Enu/(1 + beta))

        return self.I_nonfact(np.array([xLR, yLR, zLR, rLR, qLR, vLR]), El=El, Enu=Enu)
    
    def dsigma_NF_total(self, El, Enu):
        cL_sq_eff = self.cL_nul**2
        cR_sq_eff = self.cR**2
        cLcR_eff = self.cL_nul * self.cR
        dsigma = 0.0
        prefactor = (ALPHA/4/np.pi**4) * M_E * Enu
        if self.cp == 1:
            dsigma = prefactor * (cL_sq_eff*self.IL_nonfact(El, Enu) + cR_sq_eff * self.IR_nonfact(El, Enu) \
                                    + cLcR_eff*self.ILR_nonfact(El, Enu))
        elif self.cp == -1:
            dsigma = prefactor * (cL_sq_eff * self.IR_nonfact(El, Enu) + cR_sq_eff*self.IL_nonfact(El, Enu) \
                                    + cLcR_eff*self.ILR_nonfact(El, Enu))
        
        return dsigma

    def dsigma_dEl_LO(self, El, Enu, cL_sq_eff=None, cR_sq_eff=None, cLcR_eff=None):
        if cL_sq_eff is None:
            cL_sq_eff = self.cL_nul**2
        if cR_sq_eff is None:
            cR_sq_eff = self.cR**2
        if cLcR_eff is None:
            cLcR_eff = self.cL_nul * self.cR
        dsigma = 0.0
        if self.cp == 1:
            dsigma = self.xs_prefactor * (cL_sq_eff + cR_sq_eff * self.IR(El, Enu) \
                                          + cLcR_eff*self.ILR(El, Enu))
        elif self.cp == -1:
            dsigma = self.xs_prefactor * (cL_sq_eff * self.IR(El, Enu) + cR_sq_eff \
                                          + cLcR_eff*self.ILR(El, Enu))
        
        return dsigma
    
    def dsigma_dEl_LO_running(self, El, Enu):
        dsigma = 0.0
        if self.cp == 1:
            dsigma = self.xs_prefactor * (self.cL_nul_running(El)**2 + self.cR_running(El)**2 * self.IR(El, Enu) \
                                          + self.cL_nul_running(El)*self.cR_running(El)*self.ILR(El, Enu))
        elif self.cp == -1:
            dsigma = self.xs_prefactor * (self.cL_nul_running(El)**2 * self.IR(El, Enu) + self.cR_running(El)**2 \
                                          + self.cL_nul_running(El)*self.cR_running(El)*self.ILR(El, Enu))
        
        return dsigma

    def dsigma_dEl_NLO(self, El, Enu, mu=2000.0):
        beta = np.sqrt(1 - np.power(M_E/El, 2))
        EnuPrime = M_E + Enu - El

        deltas = (ALPHA/np.pi) * (self.deltav(beta) + self.delta_soft(beta) \
                                + self.deltaI(beta, Enu, EnuPrime) \
                                + self.deltaII(beta, Enu, EnuPrime))
        
        dsigma_lo_corr = (1 + deltas)*self.dsigma_dEl_LO(El, Enu)

        return dsigma_lo_corr + self.dsigma_NF_total(El, Enu) + self.dsigma_dyn(El, Enu, mu)




