from muon_decay import *
from nu_crosssection import *

import sys
sys.path.append("../")
from alplib.materials import *

from tqdm import tqdm
import vegas

class NeutrinoFluxMuonRing:
    def __init__(self, Emu=500.0e3, ring_radius=1000.0, N_muons=2.16e20, det_dist=200.0, det_area=25.0,
                 det_length=10.0, det_mat = Material("Ar"), n_samples=1000000) -> None:
        """
        Takes in the muon beam energy (MeV)
        Ring radius (km)
        Number of muon decays in the total exposure
        Detector distance from the ring tangent, area, and length (meters)
        Detector matieral class, e.g. Material("Ar") for a liquid argon detector
        n_samples: number of weighted monte carlo samples to draw
        """
        self.Emu=Emu
        self.R = ring_radius * 1000.0  # km -> m
        self.Nmu = N_muons
        self.det_dist = det_dist
        self.det_area = det_area
        self.det_width = np.sqrt(det_area)
        self.det_length = det_length
        self.decay_segment = arctan(sqrt(det_area)/det_dist)/np.pi  # fraction of circumference
        self.det_mat = det_mat
        self.n_samples = n_samples

        self.flux_weights_nue = []
        self.flux_weights_numu = []
        self.flux_energies = []
        self.flux_thetas = []

    def simulate_fluxes_mc(self, lower_y=1e-4):
        # Clear vars
        self.flux_weights_nue = []
        self.flux_weights_numu = []
        self.flux_energies = []
        self.flux_thetas = []

        self.pnumu_3vec_lab = None
        self.pnuebar_3vec_lab = None
        self.pnumubar_3vec_lab = None
        self.pnue_3vec_lab = None

        self.x_numu_at_L = None
        self.y_numu_at_L = None
        self.x_numubar_at_L = None
        self.y_numubar_at_L = None
        self.x_nue_at_L = None
        self.y_nue_at_L = None
        self.x_nuebar_at_L = None
        self.y_nuebar_at_L = None

        max_cos_numu = np.clip((-3*M_MU**2 + 4*self.Emu**2 * lower_y) / (4*self.Emu**2 *lower_y * sqrt(1 - power(M_MU/self.Emu,2))),
                          a_min=-1.0, a_max=1.0)
        max_cos_nuebar = np.clip((-M_MU**2 + 2*self.Emu**2 * lower_y) / (2*self.Emu**2 *lower_y * sqrt(1 - power(M_MU/self.Emu,2))),
                          a_min=-1.0, a_max=1.0)
        theta_rnd_numu = np.exp(np.random.uniform(log(1e-8), log(np.arccos(max_cos_numu)), self.n_samples))
        theta_rnd_nuebar = np.exp(np.random.uniform(log(1e-8), log(np.arccos(max_cos_nuebar)), self.n_samples))
        phi_rnd = np.random.uniform(0.0, 2*pi, self.n_samples)
        Enu_rnd = np.exp(np.random.uniform(log(self.Emu*lower_y), log(self.Emu), self.n_samples))

        # Construct 3-vectors for the neutrinos
        pnumu_3vec = np.array([Enu_rnd * np.sin(theta_rnd_numu) * np.cos(phi_rnd),
                             Enu_rnd * np.sin(theta_rnd_numu) * np.sin(phi_rnd),
                             Enu_rnd * np.cos(theta_rnd_numu)]).transpose()
        pnuebar_3vec = np.array([Enu_rnd * np.sin(theta_rnd_nuebar) * np.cos(phi_rnd),
                             Enu_rnd * np.sin(theta_rnd_nuebar) * np.sin(phi_rnd),
                             Enu_rnd * np.cos(theta_rnd_nuebar)]).transpose()
        
        #pnu_3vec = np.array([np.zeros(self.n_samples),
        #                     np.zeros(self.n_samples),
        #                     Enu_rnd*np.ones(self.n_samples)]).transpose()

        # Draw random anglues around the storage ring for decay positionns
        theta_intersect = np.arcsin(self.det_dist/self.R) if self.det_dist < self.R else pi/2
        theta_ring_muon = np.random.uniform(-pi/2, theta_intersect, self.n_samples)
        theta_ring_antimuon = np.random.uniform(theta_intersect, pi, self.n_samples)

        Rmatrix = np.array([[[1.0, 0.0, 0.0],
                            [0.0, cos(theta), -sin(theta)],
                            [0.0, sin(theta), cos(theta)]] for theta in theta_ring_muon])
        
        Rmatrix_conj = np.array([[[1.0, 0.0, 0.0],
                            [0.0, cos(theta), -sin(theta)],
                            [0.0, sin(theta), cos(theta)]] for theta in theta_ring_antimuon])

        # Rotate the 3-vectors around the ring about the x-axis
        # matrix multiplies R_k,ij * p_k,i --> p_k,j to get new k-array of 3-vectors)
        #pnu_3vec_lab = np.einsum('kij,ki->kj', Rmatrix, pnu_3vec)
        #pnu_conj_3vec_lab = -np.einsum('kij,ki->kj', Rmatrix_conj, pnu_3vec)  # conjugate particles in opposite circulation

        pnumu_3vec_lab = np.array([Rmatrix[i] @ pnumu_3vec[i] for i in range(self.n_samples)])
        pnuebar_3vec_lab = np.array([Rmatrix[i] @ pnuebar_3vec[i] for i in range(self.n_samples)])
        pnumubar_3vec_lab = np.array([Rmatrix_conj[i] @ (-pnumu_3vec[i]) for i in range(self.n_samples)])
        pnue_3vec_lab = np.array([Rmatrix_conj[i] @ (-pnuebar_3vec[i]) for i in range(self.n_samples)])

        # Get the projection onto the z=L (x,y) plane for numu and nuebar: parameter t for the ray trace
        t_zL_numu_is_positive = ((self.det_dist - self.R*sin(theta_ring_muon))/pnumu_3vec_lab[:,2]) > 0.0
        t_y0_numu_is_positive = ((self.R - self.R*cos(theta_ring_muon))/pnumu_3vec_lab[:,1]) > 0.0
        t_zL_numubar_is_positive = ((self.det_dist - self.R*sin(theta_ring_antimuon))/pnumubar_3vec_lab[:,2]) > 0.0
        t_y0_numubar_is_positive = ((self.R - self.R*cos(theta_ring_antimuon))/pnumubar_3vec_lab[:,1]) > 0.0

        t_zL_nuebar_is_positive = ((self.det_dist - self.R*sin(theta_ring_muon))/pnuebar_3vec_lab[:,2]) > 0.0
        t_y0_nuebar_is_positive = ((self.R - self.R*cos(theta_ring_muon))/pnuebar_3vec_lab[:,1]) > 0.0
        t_zL_nue_is_positive = ((self.det_dist - self.R*sin(theta_ring_antimuon))/pnue_3vec_lab[:,2]) > 0.0
        t_y0_nue_is_positive = ((self.R - self.R*cos(theta_ring_antimuon))/pnue_3vec_lab[:,1]) > 0.0

        # Nu mu intersections at detector plane
        self.x_numu_at_L = (pnumu_3vec_lab[:,0]/pnumu_3vec_lab[:,2]) * (self.det_dist - self.R*sin(theta_ring_muon))
        self.y_numu_at_L = self.R * (cos(theta_ring_muon) - 1) \
            + (pnumu_3vec_lab[:,1]/pnumu_3vec_lab[:,2]) * (self.det_dist - self.R*sin(theta_ring_muon))
        self.x_numu_at_y0 = (pnumu_3vec_lab[:,0]/pnumu_3vec_lab[:,1]) * self.R * (1 - cos(theta_ring_muon))
        self.z_numu_at_y0 = self.R*sin(theta_ring_muon) + (pnumu_3vec_lab[:,2]/pnumu_3vec_lab[:,1]) * self.R * (1 - cos(theta_ring_muon))

        # Numubar intersections at detector plane
        self.x_numubar_at_L = (pnumubar_3vec_lab[:,0]/pnumubar_3vec_lab[:,2]) * (self.det_dist - self.R*sin(theta_ring_antimuon))
        self.y_numubar_at_L = self.R * (cos(theta_ring_antimuon) - 1) \
            + (pnumubar_3vec_lab[:,1]/pnumubar_3vec_lab[:,2]) * (self.det_dist - self.R*sin(theta_ring_antimuon))
        self.x_numubar_at_y0 = (pnumubar_3vec_lab[:,0]/pnumubar_3vec_lab[:,1]) * self.R * (1 - cos(theta_ring_antimuon))
        self.z_numubar_at_y0 = self.R*sin(theta_ring_antimuon) + (pnumubar_3vec_lab[:,2]/pnumubar_3vec_lab[:,1]) * self.R * (1 - cos(theta_ring_antimuon))

        # Nuebar intersections at detector plane
        self.x_nuebar_at_L = (pnuebar_3vec_lab[:,0]/pnuebar_3vec_lab[:,2]) * (self.det_dist - self.R*sin(theta_ring_muon))
        self.y_nuebar_at_L = self.R * (cos(theta_ring_muon) - 1) \
            + (pnuebar_3vec_lab[:,1]/pnuebar_3vec_lab[:,2]) * (self.det_dist - self.R*sin(theta_ring_muon))
        self.x_nuebar_at_y0 = (pnuebar_3vec_lab[:,0]/pnuebar_3vec_lab[:,1]) * self.R * (1 - cos(theta_ring_muon))
        self.z_nuebar_at_y0 = self.R*sin(theta_ring_muon) + (pnuebar_3vec_lab[:,2]/pnuebar_3vec_lab[:,1]) * self.R * (1 - cos(theta_ring_muon))

        # Nue intersections at detector plane
        self.x_nue_at_L = (pnue_3vec_lab[:,0]/pnue_3vec_lab[:,2]) * (self.det_dist - self.R*sin(theta_ring_antimuon))
        self.y_nue_at_L = self.R * (cos(theta_ring_antimuon) - 1) \
            + (pnue_3vec_lab[:,1]/pnue_3vec_lab[:,2]) * (self.det_dist - self.R*sin(theta_ring_antimuon))
        self.x_nue_at_y0 = (pnue_3vec_lab[:,0]/pnue_3vec_lab[:,1]) * self.R * (1 - cos(theta_ring_antimuon))
        self.z_nue_at_y0 = self.R*sin(theta_ring_antimuon) + (pnue_3vec_lab[:,2]/pnue_3vec_lab[:,1]) * self.R * (1 - cos(theta_ring_antimuon))

        # Check intersection exists:
        # we don't care about x as x distribution is highly planar
        numu_in_det = ((self.z_numu_at_y0 < (self.det_dist + self.det_length/2)) & (self.z_numu_at_y0 > (self.det_dist-self.det_length/2))) \
                    | ((self.y_numu_at_L < self.det_width/2) & (self.y_numu_at_L > -self.det_width/2))
        numubar_in_det = ((self.z_numubar_at_y0 < (self.det_dist + self.det_length/2)) & (self.z_numubar_at_y0 > (self.det_dist-self.det_length/2))) \
                    | ((self.y_numubar_at_L < self.det_width/2) & (self.y_numubar_at_L > -self.det_width/2))
        nue_in_det = ((self.z_nue_at_y0 < (self.det_dist + self.det_length/2)) & (self.z_nue_at_y0 > (self.det_dist-self.det_length/2))) \
                    | ((self.y_nue_at_L < self.det_width/2) & (self.y_nue_at_L > -self.det_width/2))
        nuebar_in_det = ((self.z_nuebar_at_y0 < (self.det_dist + self.det_length/2)) & (self.z_nuebar_at_y0 > (self.det_dist-self.det_length/2))) \
                    | ((self.y_nuebar_at_L < self.det_width/2) & (self.y_nuebar_at_L > -self.det_width/2))
        
        flux_wgts_numu = 2 * pi * sin(theta_rnd_numu) * \
            (1/self.Emu)*d2NdydOmega_numu(Enu_rnd/self.Emu, theta_rnd_numu, self.Emu, 1.0, self.Nmu)
        
        flux_wgts_nue = 2 * pi * sin(theta_rnd_nuebar) * \
            (1/self.Emu)*d2NdydOmega_nue(Enu_rnd/self.Emu, theta_rnd_nuebar, self.Emu, 1.0, self.Nmu)

        mc_wgt_numu = theta_rnd_numu*Enu_rnd*log(np.arccos(max_cos_numu)/1e-8)*log(1/lower_y)/(self.n_samples)
        mc_wgt_nue = theta_rnd_nuebar*Enu_rnd*log(np.arccos(max_cos_nuebar)/1e-8)*log(1/lower_y)/(self.n_samples)

        self.flux_energies = Enu_rnd
        # Independent of geometry weights - need to be multiplied by accceptance weight afterwards
        self.flux_weights_numu = flux_wgts_numu * mc_wgt_numu
        self.flux_weights_nue = flux_wgts_nue * mc_wgt_nue

        self.accept_wgt_nue = flux_wgts_nue * mc_wgt_nue * nue_in_det * (pi - theta_intersect) / (2*pi)
        self.accept_wgt_numu = flux_wgts_numu * mc_wgt_numu * numu_in_det * (theta_intersect + pi/2) / (2*pi)
        self.accept_wgt_nuebar = flux_wgts_nue * mc_wgt_nue * nuebar_in_det * (theta_intersect + pi/2) / (2*pi)
        self.accept_wgt_numubar = flux_wgts_numu * mc_wgt_numu * numubar_in_det * (pi - theta_intersect) / (2*pi)

        self.intersect_wgt_numu = np.logical_or(t_zL_numu_is_positive, t_y0_numu_is_positive)
        self.intersect_wgt_nuebar = np.logical_or(t_zL_nuebar_is_positive, t_y0_nuebar_is_positive)
        self.intersect_wgt_numubar = np.logical_or(t_zL_numubar_is_positive, t_y0_numubar_is_positive)
        self.intersect_wgt_nue = np.logical_or(t_zL_nue_is_positive, t_y0_nue_is_positive)

        self.pnumu_3vec_lab = pnumu_3vec_lab
        self.pnuebar_3vec_lab = pnuebar_3vec_lab
        self.pnumubar_3vec_lab = pnumubar_3vec_lab
        self.pnue_3vec_lab = pnue_3vec_lab

        self.theta_ring_muon = theta_ring_muon
        self.theta_ring_antimuon = theta_ring_antimuon
    
    def simulate_flux(self, n_samples=100000):
        # Simulates the flux assuming approximate fraction of captured flux
        # using the decay segment fraction of ring circumference
        # Clear vars
        self.flux_weights_nue = []
        self.flux_weights_numu = []
        self.flux_energies = []
        self.flux_thetas = []

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
                 weak_mixing_angle_squared=SSW, ssw_running=False, energy_only_flux=False):
        """
        Takes in a neutrino flux of [energy (MeV), theta(rad), count]
        The flux is already integrated over the detector area
        Detector specifications: Material and Length in meters
        Weak Mixing Angle squared: specify the fixed Sin^2(theta_W), if left as none, it will
                                   take the value passed to simulate_eves()
        """
        self.energy_only_flux = energy_only_flux
        if energy_only_flux:
            self.nu_flux_energies = nu_flux[:,0]
            self.nu_flux_weights = nu_flux[:,1]
        else:
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
            if self.energy_only_flux:
                Enu = nu[0]
                theta_nu = 0.0
                wgt = nu[1]
            else:
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
            if self.energy_only_flux:
                Enu = nu[0]
                theta_nu = 0.0
                wgt = nu[1]
            else:
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
        q0 in MeV; the average momentum transfer scale
        sw2_deriv; the derivative of sin^2(theta_W)(q) to first order (in units of per MeV)
        sw20; the value of sw2 at q0
        """
        delta = "e" in flavor
        prefactor = 2 * G_F**2 * M_E / pi
        if true_running:
            q = sqrt(2*M_E*Er)
            sw2 = sw2_running(1e-3*q)  # sw2_running takes in GeV
        else:
            q = sqrt(2*M_E*Er)
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