from muon_decay import *
from MuonCollider.nu_crosssection import *

import sys
sys.path.append("../")
from alplib.materials import *

from tqdm import tqdm

class NeutronFluxMuonRing:
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
        
    def eves_event_rate(self, er_bins, n_subsamples=5):
        event_weights_numu = []
        event_weights_nue = []


        # number of targets per cm^3 * det length --> # / cm^2
        cross_section_prefactor = (AVOGADRO * self.det_mat.density / (self.det_mat.z[0] + self.det_mat.n[0])) \
            * power(HBARC, 2) * (self.det_length * 100)

        for i, Er in enumerate(tqdm(er_bins[:-1])):
            Er_low = er_bins[i]
            Er_high = er_bins[i+1]
            bin_width = Er_high - Er_low
            subsample_width = bin_width/(n_subsamples+1)

            # draw N Er subsamples between Er_low and Er_high
            Er_subsample_edges = np.linspace(Er_low, Er_high, n_subsamples+1)
            Er_subsamples = (Er_subsample_edges[1:] + Er_subsample_edges[:-1])/2

            # for each Er subsample, dblquad the flux * cross section integrand
            # TODO: integrate over detector area only
            def numu_flux_xs_integrand(Enu, theta, Er):
                return self.decay_segment * 2 * pi * sin(theta) * (1/self.Emu)*d2NdydOmega_numu(Enu/self.Emu, theta, self.Emu, 1.0, self.Nmu) \
                    * cross_section_prefactor * dsigma_dEr_eves(Er, Enu, self.sw2, flavor="mu")
            
            def nue_flux_xs_integrand(Enu, theta, Er):
                return self.decay_segment * 2 * pi * sin(theta) * (1/self.Emu)*d2NdydOmega_nue(Enu/self.Emu, theta, self.Emu, 1.0, self.Nmu) \
                    * cross_section_prefactor * dsigma_dEr_eves(Er, Enu, self.sw2, flavor="e")
            
            this_weight_nu_mu = 0.0
            this_weight_nu_e = 0.0
            for Er in Er_subsamples:
                # integrate func(y, x) from x_a to x_b, y_a to y_b
                # theta bounds: 0 to 1e-2 rad
                # Enu bounds: Enu_min(Er) to the muon energy
                theta_rnd = np.random.uniform(0.0, 1e-2, 10000)
                Enu_rnd = np.random.uniform(Enu_min(Er), self.Emu, 10000)
                mc_wgt = (1e-2)*(self.Emu - Enu_min(Er))/10000
                #integral_nu_mu = dblquad(numu_flux_xs_integrand, 0.0, 1e-2, Enu_min(Er), self.Emu, args=(Er,))[0]
                #integral_nu_e = dblquad(nue_flux_xs_integrand, 0.0, 1e-2, Enu_min(Er), self.Emu, args=(Er,))[0]

                integral_nu_mu = mc_wgt * numu_flux_xs_integrand(Enu_rnd, theta_rnd, Er)
                integral_nu_e = mc_wgt * nue_flux_xs_integrand(Enu_rnd, theta_rnd, Er)

                # rectangle rule for the Er subsample integration
                this_weight_nu_mu += np.sum(integral_nu_mu) * subsample_width
                this_weight_nu_e += np.sum(integral_nu_e) * subsample_width
            
            event_weights_numu.append(this_weight_nu_mu)
            event_weights_nue.append(this_weight_nu_e)

        return event_weights_numu, event_weights_nue
    
    def eves_weighted_mc_sim(self, er_bins, n_samples=500000, n_subsamples=10):
        self.el_weights_nue = []
        self.el_weights_numu = []
        self.el_energies = []
        self.el_thetas = []
        self.el_phis = []

        # number of targets per cm^3 * det length --> # / cm^2
        cross_section_prefactor = (AVOGADRO * self.det_mat.density / (self.det_mat.z[0] + self.det_mat.n[0])) \
            * power(HBARC, 2) * (self.det_length * 100)

        # log-spaced monte carlo
        er_rnd = np.random.uniform(er_bins[0], er_bins[-1], n_samples)

        for i, Er in enumerate(tqdm(er_rnd)):

            # draw sqrt(N) flux samples
            theta_rnd = np.exp(np.random.uniform(log(1e-6), log(1e-2), n_subsamples))
            Enu_rnd = np.exp(np.random.uniform(log(Enu_min(Er)), log(self.Emu), n_subsamples))
            phi_el_rnd = np.random.uniform(0.0, 2*pi, n_subsamples)
            mc_wgt = theta_rnd*Enu_rnd*(er_bins[-1]-er_bins[0])*log(1e-2/1e-6)*log(self.Emu/Enu_min(Er))/(n_samples*n_subsamples)

            # for each Er subsample, dblquad the flux * cross section integrand
            # TODO: integrate over detector area only
            def numu_flux_xs_integrand(Enu, theta, Er):
                return self.decay_segment * 2 * pi * sin(theta) * (1/self.Emu)*d2NdydOmega_numu(Enu/self.Emu, theta, self.Emu, 1.0, self.Nmu) \
                    * cross_section_prefactor * dsigma_dEr_eves(Er, Enu, self.sw2, flavor="mu")
            
            def nue_flux_xs_integrand(Enu, theta, Er):
                return self.decay_segment * 2 * pi * sin(theta) * (1/self.Emu)*d2NdydOmega_nue(Enu/self.Emu, theta, self.Emu, 1.0, self.Nmu) \
                    * cross_section_prefactor * dsigma_dEr_eves(Er, Enu, self.sw2, flavor="e")
            
            integral_nu_mu = mc_wgt * numu_flux_xs_integrand(Enu_rnd, theta_rnd, Er)
            integral_nu_e = mc_wgt * nue_flux_xs_integrand(Enu_rnd, theta_rnd, Er)

            # cosine of electron: cosBeta = ((Enu + me)/(Enu)) * sqrt(Er/(2me))
            theta_el = arccos(np.clip( ((Enu_rnd + M_E)/Enu_rnd) * sqrt(Er/(2*M_E + Er)), a_min=-1.0, a_max=1.0))

            # actual lab frame angle w.r.t. beam axis
            theta_z_el = arccos(cos(theta_el)*cos(theta_rnd) + cos(phi_el_rnd)*sin(theta_el)*sin(theta_rnd))

            self.el_weights_nue.extend(integral_nu_e)
            self.el_weights_numu.extend(integral_nu_mu)
            self.el_energies.extend(np.ones(n_subsamples)*Er)
            self.el_thetas.extend(theta_z_el)
            self.el_phis.extend(phi_el_rnd)
    
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

    def simulate_scatter_from_flux(self, n_samples=100, gL_mod=1.0, gR_mod=1.0, s2w=SSW):
        # simulate Er from Er_min to Er_max for each Enu

        cross_section_prefactor = (AVOGADRO * self.det_mat.density / (self.det_mat.z[0] + self.det_mat.n[0])) \
            * power(HBARC, 2) * (self.det_length * 100)
        
        flux_enu = np.array(self.flux_energies)
        theta_enu = np.array(self.flux_thetas)
        flux_weights_nue = np.array(self.flux_weights_nue)
        flux_weights_numu = np.array(self.flux_weights_numu)
        Er_max = 2*flux_enu**2 / (2*flux_enu + M_E)

        Er_rnd = np.random.uniform(0.0, Er_max, size=(n_samples, Er_max.shape[0]))
        Enu_tiled = np.tile(flux_enu, n_samples)
        Theta_nu_tiled = np.tile(theta_enu, n_samples)
        wgts_nue_tiled = np.tile(flux_weights_nue, n_samples)
        wgts_numu_tiled = np.tile(flux_weights_numu, n_samples)

        




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
    