from muon_decay import *
from nu_crosssection import *

import sys
sys.path.append("../")
from alplib.materials import *

from tqdm import tqdm

class NeutrinoFluxMuonRing:
    def __init__(self, Emu=500.0e3, ring_radius=1000.0, N_muons=2.16e20, det_dist=200.0, det_area=25.0,
                 det_length=10.0, det_mat = Material("Ar"), n_samples=1000000,
                 sw2 = SSW) -> None:
        self.Emu=Emu
        self.R = ring_radius
        self.Nmu = N_muons
        self.det_dist = det_dist
        self.det_area = det_area
        self.det_length = det_length
        self.decay_segment = 2*arctan(sqrt(det_area)/det_dist)  # fraction of circumference
        self.det_mat = det_mat
        self.n_samples = n_samples

        self.el_weights_nue = []
        self.el_weights_numu = []
        self.el_energies = []
        self.el_thetas = []
        self.el_phis = []

        self.flux_weights_nue = []
        self.flux_weights_numu = []
        self.flux_energies = []
        self.flux_thetas = []

        # parameters
        self.sw2 = sw2
    
    def simulate_fluxes_mc(self):
        y_rnd = np.random.uniform(0, 1, self.n_samples)
        theta_rnd = np.random.uniform(0.0, 1e-2, self.n_samples)

        # simulate random decay points along the valid ring segment
        # deviation of angles is roughly linear

        dtheta = 2*pi*self.decay_segment * np.random.uniform(-1, 1, self.n_samples)
        lab_angles = abs(theta_rnd + dtheta)
        
        diff_flux_mu = d2NdydOmega_numu(y_rnd, theta_rnd, self.Emu, self.det_dist, self.Nmu)
        diff_flux_e = d2NdydOmega_nue(y_rnd, theta_rnd, self.Emu, self.det_dist, self.Nmu)
        #diff_flux_mubar = d2NdydOmega_numu(y_rnd, theta_rnd, self.Emu, self.det_dist, self.Nmu)
        #diff_flux_ebar = d2NdydOmega_nue(y_rnd, theta_rnd, self.Emu, self.det_dist, self.Nmu)

        mc_wgt = self.decay_segment * 2*pi*sin(theta_rnd)*(1e-2)/self.n_samples  # per m^2 per year

        self.flux_energies_e = y_rnd * self.Emu
        self.flux_energies_mu = y_rnd * self.Emu
        self.flux_theta_e = lab_angles
        self.flux_theta_e = lab_angles
        self.flux_weights_mu = diff_flux_mu * mc_wgt  # counts / m^2
        self.flux_weights_e = diff_flux_e * mc_wgt
    
    def simulate_flux(self, n_samples=100000):
        theta_rnd = np.exp(np.random.uniform(log(1e-8), log(1e-2), n_samples))
        Enu_rnd = np.exp(np.random.uniform(log(self.Emu*1e-4), log(self.Emu), n_samples))

        mc_wgt = theta_rnd*Enu_rnd*log(1e-2/1e-8)*log(1e4)/(n_samples)

        flux_wgts_numu = self.decay_segment * 2 * pi * sin(theta_rnd) * \
            (1/self.Emu)*d2NdydOmega_numu(Enu_rnd/self.Emu, theta_rnd, self.Emu, 1.0, self.Nmu)
        
        flux_wgts_nue = self.decay_segment * 2 * pi * sin(theta_rnd) * \
            (1/self.Emu)*d2NdydOmega_nue(Enu_rnd/self.Emu, theta_rnd, self.Emu, 1.0, self.Nmu)
        
        self.flux_energies.extend(Enu_rnd)
        self.flux_thetas.extend(theta_rnd)
        self.flux_weights_nue.extend(flux_wgts_nue*mc_wgt)
        self.flux_weights_numu.extend(flux_wgts_numu*mc_wgt)




class EvESFromNeutrinoFlux:
    def __init__(self, nu_flux, flavor="mu", detector_material=Material("Ar"), detector_length=10.0,
                 weak_mixing_angle_squared=SSW, ssw_running=False):
        """
        Takes in a neutrino flux of [energy (MeV), theta(rad), count]
        The flux is already integrated over the detector area
        Detector specifications: Material and Length in meters
        Weak Mixing Angle squared: specify the fixed Sin^2(theta_W), if left as none, it will
                                   take the value passed to simulate_eves()
        """
        self.nu_flux_energies = nu_flux[:,0]
        self.nu_flux_thetas = nu_flux[:,1]
        self.nu_flux_weights = nu_flux[:,2]
        self.nu_flux = nu_flux
        self.flavor = flavor

        self.el_weights = []
        self.el_energies = []
        self.el_thetas = []
        self.el_phis = []
        self.el_parent_nu_energies = []

        self.sw2 = weak_mixing_angle_squared

        self.det_mat = detector_material
        self.det_length = detector_length

        self.running = ssw_running
    
    def simulate_eves(self, Er_min=0.0, n_samples=100, verbose=False, gL_mod=1.0, gR_mod=1.0):
        self.el_weights = []
        self.el_energies = []
        self.el_thetas = []
        self.el_phis = []
        self.el_parent_nu_energies = []


        # number of targets per cm^3 * det length --> # / cm^2
        cross_section_prefactor = (AVOGADRO * self.det_mat.density / (self.det_mat.z[0] + self.det_mat.n[0])) \
            * power(HBARC, 2) * (self.det_length * 100)
        
        # For each neutrino with energy Enu and angle theta_nu, simulate an Er spectrum
        # from Er = 0 to Er_max = 2 Enu**2 / (2 Enu + me)
        if verbose:
            print("Simulating Neutrino EvES scattering from flux...")
        for i, nu in enumerate(self.nu_flux):
            Enu = nu[0]
            theta_nu = nu[1]
            wgt = nu[2]

            Er_max = 2*Enu**2 / (2*Enu + M_E)

            # draw sqrt(N) flux samples
            Er_rnd = np.random.uniform(Er_min, Er_max, n_samples)
            phi_el_rnd = np.random.uniform(0.0, 2*pi, n_samples)
            mc_wgt = (Er_max - Er_min)/n_samples

            # for each Er subsample, dblquad the flux * cross section integrand
            xs_weights = cross_section_prefactor * mc_wgt * dsigma_dEr_eves(Er_rnd, Enu, self.sw2,
                                                                   flavor=self.flavor, gL_mod=gL_mod, gR_mod=gR_mod,
                                                                   running=self.running)
            
            # cosine of electron: cosBeta = ((Enu + me)/(Enu)) * sqrt(Er/(2me))
            theta_el = arccos(np.clip( ((Enu + M_E)/Enu) * sqrt(Er_rnd/(2*M_E + Er_rnd)), a_min=-1.0, a_max=1.0))

            # actual lab frame angle w.r.t. beam axis
            theta_z_el = arccos(cos(theta_el)*cos(theta_nu) + cos(phi_el_rnd)*sin(theta_el)*sin(theta_nu))

            self.el_weights.extend(wgt*xs_weights)
            self.el_energies.extend(Er_rnd)
            self.el_thetas.extend(theta_z_el)
            self.el_phis.extend(phi_el_rnd)
            self.el_parent_nu_energies.extend(np.ones(n_samples)*Enu)
    
    def simulate_eves_rectangle_rule(self, Er_min=0.0, n_samples=1000, verbose=False,
                                     gL_mod=1.0, gR_mod=1.0, turn_off_cross_term=False):
        self.el_weights = []
        self.el_energies = []
        self.el_thetas = []
        self.el_phis = []
        self.el_parent_nu_energies = []


        # number of targets per cm^3 * det length --> # / cm^2
        cross_section_prefactor = (AVOGADRO * self.det_mat.density / (self.det_mat.z[0] + self.det_mat.n[0])) \
            * power(HBARC, 2) * (self.det_length * 100)
        
        phi_el_rnd = np.random.uniform(0.0, 2*pi, n_samples)

        # For each neutrino with energy Enu and angle theta_nu, simulate an Er spectrum
        # from Er = 0 to Er_max = 2 Enu**2 / (2 Enu + me)
        if verbose:
            print("Simulating Neutrino EvES scattering from flux...")
        for i, nu in enumerate(self.nu_flux):
            Enu = nu[0]
            theta_nu = nu[1]
            wgt = nu[2]

            Er_max = 2*Enu**2 / (2*Enu + M_E)

            # draw sqrt(N) flux samples
            Er_rnd = np.linspace(Er_min, Er_max, n_samples)
            mc_wgt = (Er_max - Er_min)/n_samples

            # for each Er subsample, dblquad the flux * cross section integrand
            if turn_off_cross_term:
                xs_weights = cross_section_prefactor * mc_wgt * dsigma_dEr_eves_noME(Er_rnd, Enu, self.sw2,
                                                                   flavor=self.flavor, gL_mod=gL_mod, gR_mod=gR_mod,
                                                                   running=self.running)
            else:
                xs_weights = cross_section_prefactor * mc_wgt * dsigma_dEr_eves(Er_rnd, Enu, self.sw2,
                                                                   flavor=self.flavor, gL_mod=gL_mod, gR_mod=gR_mod,
                                                                   running=self.running)
            
            # cosine of electron: cosBeta = ((Enu + me)/(Enu)) * sqrt(Er/(2me))
            theta_el = arccos(np.clip( ((Enu + M_E)/Enu) * sqrt(Er_rnd/(2*M_E + Er_rnd)), a_min=-1.0, a_max=1.0))

            # actual lab frame angle w.r.t. beam axis
            theta_z_el = arccos(cos(theta_el)*cos(theta_nu) + cos(phi_el_rnd)*sin(theta_el)*sin(theta_nu))

            self.el_weights.extend(wgt*xs_weights)
            self.el_energies.extend(Er_rnd)
            self.el_thetas.extend(theta_z_el)
            self.el_phis.extend(phi_el_rnd)
            self.el_parent_nu_energies.extend(np.ones(n_samples)*Enu)
    
    def get_energy_theta_phi_wgt_arrays(self):
        return np.array(self.el_energies), np.array(self.el_thetas), \
                np.array(self.el_phis), np.array(self.el_weights)




class EvESFromNeutrinoFluxRunningSSW(EvESFromNeutrinoFlux):
    def __init__(self, nu_flux, flavor="mu", detector_material=Material("Ar"), detector_length=10.0,
                 weak_mixing_angle_squared=SSW):
        super().__init__(nu_flux, flavor, detector_material, detector_length, weak_mixing_angle_squared, True)

    def dsigma_dEr_eves_running(self, Er, Enu, sw20, sw2_deriv, q0, flavor="mu", true_running=True):
        """
        Takes in flavors "e", "mu", "tau", "ebar", "mubar", "taubar"
        """
        delta = "e" in flavor
        prefactor = 2 * G_F**2 * M_E / pi
        if true_running:
            q = 1e-3*sqrt(2*M_E*Er)
            sw2 = sw2_running(q)
        else:
            q = 1e-3*sqrt(2*M_E*Er)
            sw2 = sw20 + sw2_deriv*(q - q0)
        gL = delta + (sw2 - 0.5)
        gR = sw2
        if "bar" in flavor:
            return prefactor*((gR)**2 + power(gL * (1-Er/Enu), 2) - gL*gR*M_E*Er/power(Enu,2))

        return prefactor*((gL)**2 + power(gR * (1-Er/Enu), 2) - gL*gR*M_E*Er/power(Enu,2))

    def simulate_eves_rectangle_rule(self, Er_min=0.0, n_samples=1000, verbose=False,
                                     sw20=SSW, sw2_deriv=0.0, q0=0.0, true_running=True):
        self.el_weights = []
        self.el_energies = []
        self.el_thetas = []
        self.el_phis = []
        self.el_parent_nu_energies = []


        # number of targets per cm^3 * det length --> # / cm^2
        cross_section_prefactor = (AVOGADRO * self.det_mat.density / (self.det_mat.z[0] + self.det_mat.n[0])) \
            * power(HBARC, 2) * (self.det_length * 100)
        
        phi_el_rnd = np.random.uniform(0.0, 2*pi, n_samples)

        # For each neutrino with energy Enu and angle theta_nu, simulate an Er spectrum
        # from Er = 0 to Er_max = 2 Enu**2 / (2 Enu + me)
        if verbose:
            print("Simulating Neutrino EvES scattering from flux...")
        for i, nu in enumerate(self.nu_flux):
            Enu = nu[0]
            theta_nu = nu[1]
            wgt = nu[2]

            Er_max = 2*Enu**2 / (2*Enu + M_E)

            # draw sqrt(N) flux samples
            Er_rnd = np.linspace(Er_min, Er_max, n_samples)
            mc_wgt = (Er_max - Er_min)/n_samples

            # for each Er subsample, dblquad the flux * cross section integrand

            xs_weights = cross_section_prefactor * mc_wgt * self.dsigma_dEr_eves_running(Er_rnd, Enu,
                                                                sw20, sw2_deriv, q0,
                                                                flavor=self.flavor, true_running=true_running)
        
            # cosine of electron: cosBeta = ((Enu + me)/(Enu)) * sqrt(Er/(2me))
            theta_el = arccos(np.clip( ((Enu + M_E)/Enu) * sqrt(Er_rnd/(2*M_E + Er_rnd)), a_min=-1.0, a_max=1.0))

            # actual lab frame angle w.r.t. beam axis
            theta_z_el = arccos(cos(theta_el)*cos(theta_nu) + cos(phi_el_rnd)*sin(theta_el)*sin(theta_nu))

            self.el_weights.extend(wgt*xs_weights)
            self.el_energies.extend(Er_rnd)
            self.el_thetas.extend(theta_z_el)
            self.el_phis.extend(phi_el_rnd)
            self.el_parent_nu_energies.extend(np.ones(n_samples)*Enu)