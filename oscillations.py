import sys
sys.path.append('../')
from alplib.constants import *
import numpy as np

def oscillation_parameters(t12=0.5763617589722192,
                           t13=0.14819001778459273,
                           t23=0.7222302630963306,
                           delta=1.5*np.pi,
                           d21=7.37e-17,
                           d31=2.5e-15+3.685e-17):
    r"""
    creating a list of oscillation parameter, default: LMA solution
    :param t12: \theta_12
    :param t23: \theta_23
    :param t13: \theta_13
    :param delta: \delta
    :param d21: \Delta m^{2}_{21} in MeV^2
    :param d31: \Delta m^{2}_{31} in MeV^2
    :return: list of oscillation parameter
    """
    return {'t12': t12, 't13': t13, 't23': t23, 'delta': delta, 'd21': d21, 'd31': d31}



class NSIparameters:
    r"""
    nsi parameter class,
    g = g_\nu*g_f
    mz = mediator mass
    for scattering, it is more convenient to use g
    for oscillation, it is more convenient to use epsilon
    it contains L and R couplings of electron scattering,
    and vector couplings of quarks
    """
    def __init__(self, mz=0, mphi=0):
        """
        initializing all nsi == 0
        """

        # Vector NSI
        self.mz = mz
        self.gel = {'ee': 0, 'mm': 0, 'tt': 0, 'em': 0, 'et': 0, 'mt': 0}
        self.ger = {'ee': 0, 'mm': 0, 'tt': 0, 'em': 0, 'et': 0, 'mt': 0}
        self.gu = {'ee': 0, 'mm': 0, 'tt': 0, 'em': 0, 'et': 0, 'mt': 0}
        self.gd = {'ee': 0, 'mm': 0, 'tt': 0, 'em': 0, 'et': 0, 'mt': 0}
        self.epel = {'ee': 0, 'mm': 0, 'tt': 0, 'em': 0, 'et': 0, 'mt': 0}
        self.eper = {'ee': 0, 'mm': 0, 'tt': 0, 'em': 0, 'et': 0, 'mt': 0}
        self.epe = {'ee': 0, 'mm': 0, 'tt': 0, 'em': 0, 'et': 0, 'mt': 0}
        self.epu = {'ee': 0, 'mm': 0, 'tt': 0, 'em': 0, 'et': 0, 'mt': 0}
        self.epd = {'ee': 0, 'mm': 0, 'tt': 0, 'em': 0, 'et': 0, 'mt': 0}

        # Scalar NSI
        # eta = y_f * y_ab / m_phi^2
        # y: yukawas
        # m_phi: scalar mass
        self.mphi = mphi
        self.eta_e = {'ee': 0, 'mm': 0, 'tt': 0, 'em': 0, 'et': 0, 'mt': 0}
        self.eta_u = {'ee': 0, 'mm': 0, 'tt': 0, 'em': 0, 'et': 0, 'mt': 0}
        self.eta_d = {'ee': 0, 'mm': 0, 'tt': 0, 'em': 0, 'et': 0, 'mt': 0}

    def ee(self):
        """
        :return: matrix of nsi for electron
        """
        if self.mz != 0:
            for i in self.epe:
                self.epe[i] = (self.gel[i]+self.ger[i]) / (2*np.sqrt(2)*gf*self.mz**2)
        return np.array([[self.epe['ee'], self.epe['em'], self.epe['et']],
                        [np.conj(self.epe['em']), self.epe['mm'], self.epe['mt']],
                        [np.conj(self.epe['et']), np.conj(self.epe['mt']), self.epe['tt']]])

    def eu(self):
        """
        :return: matrix of nsi for u quark
        """
        if self.mz != 0:
            for i in self.epu:
                self.epu[i] = self.gu[i] / (2*np.sqrt(2)*gf*self.mz**2)
        return np.array([[self.epu['ee'], self.epu['em'], self.epu['et']],
                        [np.conj(self.epu['em']), self.epu['mm'], self.epu['mt']],
                        [np.conj(self.epu['et']), np.conj(self.epu['mt']), self.epu['tt']]])

    def ed(self):
        """
        :return: matrix of nsi for d quark
        """
        if self.mz != 0:
            for i in self.epd:
                self.epd[i] = self.gu[i] / (2*np.sqrt(2)*gf*self.mz**2)
        return np.array([[self.epd['ee'], self.epd['em'], self.epd['et']],
                        [np.conj(self.epd['em']), self.epd['mm'], self.epd['mt']],
                        [np.conj(self.epd['et']), np.conj(self.epd['mt']), self.epd['tt']]])
    
    def eta_el(self):
        """
        :return: matrix of nsi for electron
        """
        return np.array([[self.eta_e['ee'], self.eta_e['em'], self.eta_e['et']],
                        [np.conj(self.eta_e['em']), self.eta_e['mm'], self.eta_e['mt']],
                        [np.conj(self.eta_e['et']), np.conj(self.eta_e['mt']), self.eta_e['tt']]])

    def eta_up(self):
        """
        :return: matrix of nsi for u quark
        """
        return np.array([[self.eta_u['ee'], self.eta_u['em'], self.eta_u['et']],
                        [np.conj(self.eta_u['em']), self.eta_u['mm'], self.eta_u['mt']],
                        [np.conj(self.eta_u['et']), np.conj(self.eta_u['mt']), self.eta_u['tt']]])

    def eta_down(self):
        """
        :return: matrix of nsi for d quark
        """
        return np.array([[self.eta_d['ee'], self.eta_d['em'], self.eta_d['et']],
                        [np.conj(self.eta_d['em']), self.eta_d['mm'], self.eta_d['mt']],
                        [np.conj(self.eta_d['et']), np.conj(self.eta_d['mt']), self.eta_d['tt']]])

    def eel(self):
        if self.mz != 0:
            for i in self.epel:
                self.epel[i] = (self.gel[i]) / (2*np.sqrt(2)*gf*self.mz**2)
        return np.array([[self.epel['ee'], self.epel['em'], self.epel['et']],
                        [np.conj(self.epel['em']), self.epel['mm'], self.epel['mt']],
                        [np.conj(self.epel['et']), np.conj(self.epel['mt']), self.epel['tt']]])

    def eer(self):
        if self.mz != 0:
            for i in self.eper:
                self.eper[i] = (self.ger[i]) / (2*np.sqrt(2)*gf*self.mz**2)
        return np.array([[self.eper['ee'], self.eper['em'], self.eper['et']],
                        [np.conj(self.eper['em']), self.eper['mm'], self.eper['mt']],
                        [np.conj(self.eper['et']), np.conj(self.eper['mt']), self.eper['tt']]])




def survival_const(ev, length=0.0, epsi=NSIparameters(), op=oscillation_parameters(),
                   ne=2.2 * 6.02e23 * (100 * METER_BY_MEV) ** 3, nui='e', nuf='e'):
    """
    survival/transitional probability with constant matter density
    :param ev: nuetrino energy in MeV
    :param length: oscillation length in meters
    :param epsi: epsilons
    :param nui: initail flavor
    :param nuf: final flavor
    :param op: oscillation parameters
    :param ne: electron number density in MeV^3
    :return: survival/transitional probability
    """
    op = op.copy()
    dic = {'e': 0, 'mu': 1, 'tau': 2, 'ebar': 0, 'mubar': 1, 'taubar': 2}
    fi = dic[nui]
    ff = dic[nuf]
    length = length / METER_BY_MEV
    if nuf[-1] == 'r':
        op['delta'] = -op['delta']
    o23 = np.array([[1, 0, 0],
                   [0, np.cos(op['t23']), np.sin(op['t23'])],
                   [0, -np.sin(op['t23']), np.cos(op['t23'])]])
    u13 = np.array([[np.cos(op['t13']), 0, np.sin(op['t13']) * (np.exp(- op['delta'] * 1j))],
                   [0, 1, 0],
                   [-np.sin(op['t13'] * (np.exp(op['delta'] * 1j))), 0, np.cos(op['t13'])]])
    o12 = np.array([[np.cos(op['t12']), np.sin(op['t12']), 0],
                   [-np.sin(op['t12']), np.cos(op['t12']), 0],
                   [0, 0, 1]])
    umix = o23 @ u13 @ o12
    m = np.diag(np.array([0, op['d21'] / (2 * ev), op['d31'] / (2 * ev)])) + (epsi.eta_el() + epsi.eta_up() + epsi.eta_down())
    vf = np.sqrt(2) * G_F * ne * ((epsi.ee() + np.diag(np.array([1, 0, 0]))) + 3 * epsi.eu() + 3 * epsi.ed())
    if nuf[-1] == 'r':
        hf = umix @ m @ umix.conj().T - np.conj(vf)
    else:
        hf = umix @ m @ umix.conj().T + vf
    w, v = np.linalg.eigh(hf)
    res = 0.0
    for i in range(3):
        for j in range(3):
            theta = (w[i]-w[j]) * length
            res += v[ff, i] * np.conj(v[fi, i]) * np.conj(v[ff, j]) * v[fi, j] * (np.cos(theta) - 1j * np.sin(theta))
    return np.real(res)
