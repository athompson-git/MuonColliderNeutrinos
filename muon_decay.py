import numpy as np
from numpy import pi, cos, sin, sqrt, power, log

import sys
sys.path.append("../")

from alplib.constants import *
from scipy.integrate import quad
from scipy.special import spence


# From A. Cervera et al, 2000
# Unpolarized rates for muon and electron flavored neutrinos from muon/anti-muon decay
# Unpolarized are symmetric for particle <--> antiparticle
def d2NdydOmega_numu(y, theta, Emu, l_det, N_mu):
    beta = sqrt(1 - power(M_MU/Emu, 2))

    prefactor = 4*N_mu / (pi*l_det**2 * power(M_MU, 6))

    return np.clip(prefactor * power(Emu, 4) * power(y, 2) * (1 - beta*cos(theta)) \
        * (3 * M_MU**2 - 4*power(Emu, 2)*y*(1-beta*cos(theta))), a_min=0.0, a_max=np.inf)

def d2NdydOmega_nue(y, theta, Emu, l_det, N_mu):
    beta = sqrt(1 - power(M_MU/Emu, 2))

    prefactor = 24*N_mu / (pi*l_det**2 * power(M_MU, 6))

    return np.clip(prefactor * power(Emu, 4) * power(y, 2) * (1 - beta*cos(theta)) \
        * (M_MU**2 - 2*power(Emu, 2)*y*(1-beta*cos(theta))), a_min=0.0, a_max=np.inf)



def d2N_dTheta_dy_numu(theta, y, Emu, l_det, N_mu):
    beta = sqrt(1 - power(M_MU/Emu, 2))

    prefactor = 4*N_mu / (pi*l_det**2 * power(M_MU, 6))

    return sin(theta)*np.clip(prefactor * power(Emu, 4) * power(y, 2) * (1 - beta*cos(theta)) \
        * (3 * M_MU**2 - 4*power(Emu, 2)*y*(1-beta*cos(theta))), a_min=0.0, a_max=np.inf)


def d2N_dTheta_dy_nue(theta, y, Emu, l_det, N_mu):
    beta = sqrt(1 - power(M_MU/Emu, 2))

    prefactor = 24*N_mu / (pi*l_det**2 * power(M_MU, 6))

    return sin(theta)*np.clip(prefactor * power(Emu, 4) * power(y, 2) * (1 - beta*cos(theta)) \
        * (M_MU**2 - 2*power(Emu, 2)*y*(1-beta*cos(theta))), a_min=0.0, a_max=np.inf)

def dNdy_nue(y, Emu, l_det, N_mu, theta_max=1e-2):
    return 2*pi*quad(d2N_dTheta_dy_nue, 0.0, theta_max, args=(y, Emu, l_det, N_mu,))[0]

def dNdy_numu(y, Emu, l_det, N_mu, theta_max=1e-2):
    return 2*pi*quad(d2N_dTheta_dy_numu, 0.0, theta_max, args=(y, Emu, l_det, N_mu,))[0]




# Corrected Fluxes: Broncano, Mena 2003
class LabFrameNeutrinoFluxFromMuon:
    def __init__(self, Emu=5.0e6):
        self.Emu = Emu
        self.gamma = Emu/M_MU
        self.beta = sqrt(1 - power(Emu/M_MU, -2))

    def z(self, Enu):
        return Enu/self.Emu
    
    def spence_integral(self, x):
        # Use scipy's spence definition which is related by
        # L(x) = scipy.special.spence(1-x)
        return spence(1 - x)

    def kfunc(self, z, cosTheta):
        return power(log(1 - 2*self.gamma**2 * z*(1 - self.beta*cosTheta)), 2) \
                + 2*self.spence_integral(2*self.gamma**2 * z*(1 - self.beta*cosTheta)) \
                + 2*pi**2 / 3

    def F0numu(self, z, cosTheta):
        return 8*(power(self.Emu, 4) / power(M_MU, 6)) * z**2 * (1 - self.beta*cosTheta) \
                * (3*M_MU**2 - 4*self.Emu**2 * z * (1 - self.beta*cosTheta))

    def F0nuebar(self, z, cosTheta):
        return 48*(power(self.Emu, 4) / power(M_MU, 6)) * z**2 * (1 - self.beta*cosTheta) \
                * (M_MU**2 - 2*self.Emu**2 * z * (1 - self.beta*cosTheta))

    def J0numu(self, z, cosTheta):
        return 8*(power(self.Emu, 4) / power(M_MU, 6)) * z**2 * (1 - self.beta*cosTheta) \
                * (M_MU**2 - 4*self.Emu**2 * z * (1 - self.beta*cosTheta))

    def J0nuebar(self, z, cosTheta):
        return 48*(power(self.Emu, 4) / power(M_MU, 6)) * z**2 * (1 - self.beta*cosTheta) \
                * (M_MU**2 - 2*self.Emu**2 * z * (1 - self.beta*cosTheta))

    def F1numu(self, z, cosTheta):
        expression1 = 2*self.gamma**2 * z * (1-self.beta*cosTheta)

        return self.F0numu(z, cosTheta)*self.kfunc(z, cosTheta) \
            + ((41 - 36*expression1 + 42*expression1**2 - 16*expression1**3) * log(1 - expression1) \
                + 0.5*expression1*(82 - 153*expression1 + 86*expression1**2)) / (3*(1 - self.beta*cosTheta))

    def F1nuebar(self, z, cosTheta):
        expression1 = 2*self.gamma**2 * z * (1-self.beta*cosTheta)

        return self.F0nuebar(z, cosTheta)*self.kfunc(z, cosTheta) + 2*(1-expression1)/(1-self.beta*cosTheta) \
            * ((5 + 8*expression1 + 8*expression1**2) * log(1 - expression1) \
            + 0.5*expression1*(10 - 19*expression1))

    def J1numu(self, z, cosTheta):
        expression1 = 2*self.gamma**2 * z * (1-self.beta*cosTheta)

        return self.J0numu(z, cosTheta)*self.kfunc(z, cosTheta) \
            + ((11 - 36*expression1 + 14*expression1**2 - 16*expression1**3 + 4/expression1) \
               * log(1 - expression1) + 0.5*expression1*(-8 + 18*expression1 \
                                                         - 103*expression1**2 + 78*expression1**3)) \
            / (3*(1 - self.beta*cosTheta))

    def J1nuebar(self, z, cosTheta):
        expression1 = 2*self.gamma**2 * z * (1-self.beta*cosTheta)

        return self.J0nuebar(z, cosTheta)*self.kfunc(z, cosTheta) + 2*(1-expression1)/(1-self.beta*cosTheta) \
            * ((-3 + 12*expression1 + 8*expression1**2 + 4/expression1) * log(1 - expression1) \
            + 0.5*(8 - 2*expression1 - 15*expression1**2))

    def diff_flux_numu(self, Enu, theta, Pmu=0.0):
        # Returns d^2 N / dz dcos(theta)
        return self.F0numu(Enu/self.Emu, cos(theta)) + Pmu * self.J0numu(Enu/self.Emu, cos(theta)) * cos(theta) \
                - (ALPHA/(2*pi)) * (self.F1numu(Enu/self.Emu, cos(theta)) \
                                    + Pmu * self.J1numu(Enu/self.Emu, cos(theta)) * cos(theta))

    def diff_flux_nuebar(self, Enu, theta, Pmu=0.0):
        # Returns d^2 N / dz dcos(theta)
        return self.F0nuebar(Enu/self.Emu, cos(theta)) + Pmu * self.J0nuebar(Enu/self.Emu, cos(theta)) * cos(theta) \
                - (ALPHA/(2*pi)) * (self.F1nuebar(Enu/self.Emu, cos(theta)) \
                                    + Pmu * self.J1nuebar(Enu/self.Emu, cos(theta)) * cos(theta))