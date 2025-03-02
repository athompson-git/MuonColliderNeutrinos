import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.pylab import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

from scipy.stats import chi2
from scipy.special import gammaln

import pymultinest

from muon_collider_mc import *
from nu_crosssection import *

from flux_config import *



### 3 TEV CENTER OF MASS BEAMS
### WMA ANALYSIS



def log_pois(data, bkg, signal):
    return -2*np.sum(data * np.log(signal + bkg) - (signal + bkg) - gammaln(data + 1))

# Define energy range, samples, parameters etc.
N_SAMPLES = 500


q0 = 680.0  # for sqrt(s) = 3 TeV
q0_10TeV = 1500.0  # at 1.5 GeV for sqrt(s) = 10 TeV
q0_500GeV = 278.0  # for sqrt(s) = 500 GeV


# SET CENTER OF MASS ENERGY AS INPUT
COM_ENERGY = "3TeV"
MUON_ENERGY = fluxes_dict[COM_ENERGY]["Emu"]  # GeV
q_bins = fluxes_dict[COM_ENERGY]["q_bins_GeV"]  # GeV

 # Define the event generators and null hypothesis spectrum
precalc_nue_flux = fluxes_dict[COM_ENERGY]["nue_flux"]
precalc_numu_flux = fluxes_dict[COM_ENERGY]["numu_flux"]

# Generate NULL HYPOTHESIS
# sqrt(s) = 10 TeV - WITH RUNNING
eves_gen_nuebar_running = EvESFromNeutrinoFlux(nu_flux=precalc_nue_flux, flavor="ebar", detector_material=Material("Ar"),
                                    detector_length=10.0, ssw_running=True, energy_only_flux=True)
eves_gen_numu_running = EvESFromNeutrinoFlux(nu_flux=precalc_numu_flux, flavor="mu", detector_material=Material("Ar"),
                                    detector_length=10.0, ssw_running=True, energy_only_flux=True)
eves_gen_nuebar_running.simulate_eves_rectangle_rule(n_samples=N_SAMPLES)
eves_gen_numu_running.simulate_eves_rectangle_rule(n_samples=N_SAMPLES)
nuebar_energies_running, _, _, nuebar_wgts_running = eves_gen_nuebar_running.get_energy_theta_phi_wgt_arrays()
numu_energies_running, _, _, numu_wgts_running = eves_gen_numu_running.get_energy_theta_phi_wgt_arrays()

eves_gen_nue_running = EvESFromNeutrinoFlux(nu_flux=precalc_nue_flux, flavor="e", detector_material=Material("Ar"),
                                    detector_length=10.0, ssw_running=True, energy_only_flux=True)
eves_gen_numubar_running = EvESFromNeutrinoFlux(nu_flux=precalc_numu_flux, flavor="mubar", detector_material=Material("Ar"),
                                    detector_length=10.0, ssw_running=True, energy_only_flux=True)
eves_gen_nue_running.simulate_eves_rectangle_rule(n_samples=N_SAMPLES)
eves_gen_numubar_running.simulate_eves_rectangle_rule(n_samples=N_SAMPLES)
nue_energies_running, _, _, nue_wgts_running = eves_gen_nue_running.get_energy_theta_phi_wgt_arrays()
numubar_energies_running, _, _, numubar_wgts_running = eves_gen_numubar_running.get_energy_theta_phi_wgt_arrays()


h_nuebar_run, _ = np.histogram(1e-3*sqrt(2*M_E*nuebar_energies_running), weights=nuebar_wgts_running, bins=q_bins)
h_nue_run, _ = np.histogram(1e-3*sqrt(2*M_E*nue_energies_running), weights=nue_wgts_running, bins=q_bins)
h_numubar_run, _ = np.histogram(1e-3*sqrt(2*M_E*numubar_energies_running), weights=numubar_wgts_running, bins=q_bins)
h_numu_run, _ = np.histogram(1e-3*sqrt(2*M_E*numu_energies_running), weights=numu_wgts_running, bins=q_bins)


print("Setting up null hypothesis predictions...")
print("h_nuebar events = {}".format(h_nuebar_run))
print("h_numu events = {}".format(h_numu_run))
print("h_numubar events = {}".format(h_numubar_run))
print("h_nue events = {}".format(h_nue_run))


# SET UP TEST HYPOTHESIS GENERATORS
eves_gen_nuebar_testHyp = EvESFromNeutrinoFluxRunningSSW(nu_flux=precalc_nue_flux, flavor="ebar", detector_material=Material("Ar"),
                                    detector_length=10.0)
eves_gen_numu_testHyp = EvESFromNeutrinoFluxRunningSSW(nu_flux=precalc_numu_flux, flavor="mu", detector_material=Material("Ar"),
                                    detector_length=10.0)


eves_gen_nue_testHyp = EvESFromNeutrinoFluxRunningSSW(nu_flux=precalc_nue_flux, flavor="e", detector_material=Material("Ar"),
                                    detector_length=10.0)
eves_gen_numubar_testHyp = EvESFromNeutrinoFluxRunningSSW(nu_flux=precalc_numu_flux, flavor="mubar", detector_material=Material("Ar"),
                                    detector_length=10.0)



# 3 TeV
def prior_3TeV(cube, ndim, nparams):
    cube[0] = 0.21 + 0.05 * cube[0]  # SW^2 (0) constant piece
    cube[1] = 1.2e-4 * cube[1] - 1.0e-4  # dSW^2/dq (1) linear piece

# 500 GeV
def prior_500GeV(cube, ndim, nparams):
    cube[0] = 0.21 + 0.05 * cube[0]  # SW^2 (0) constant piece
    cube[1] = 1.2e-3 * cube[1] - 1.0e-3  # dSW^2/dq (1) linear piece

# 10 TeV
def prior_10TeV(cube, ndim, nparams):
    cube[0] = 0.21 + 0.05 * cube[0]  # SW^2 (0) constant piece
    cube[1] = 1.1e-4 * cube[1] - 1.0e-4  # dSW^2/dq (1) linear piece


def loglike(cube, ndim, nparams):
    a = cube[0]
    b = cube[1]
    eves_gen_nuebar_testHyp.simulate_eves_rectangle_rule(n_samples=N_SAMPLES, sw20=a, sw2_deriv=b, q0=q0, true_running=False)
    eves_gen_numu_testHyp.simulate_eves_rectangle_rule(n_samples=N_SAMPLES, sw20=a, sw2_deriv=b, q0=q0, true_running=False)
    nuebar_energies_testHyp, _, _, nuebar_wgts_testHyp = eves_gen_nuebar_testHyp.get_energy_theta_phi_wgt_arrays()
    numu_energies_testHyp, _, _, numu_wgts_testHyp = eves_gen_numu_testHyp.get_energy_theta_phi_wgt_arrays()

    eves_gen_nue_testHyp.simulate_eves_rectangle_rule(n_samples=N_SAMPLES, sw20=a, sw2_deriv=b, q0=q0, true_running=False)
    eves_gen_numubar_testHyp.simulate_eves_rectangle_rule(n_samples=N_SAMPLES, sw20=a, sw2_deriv=b, q0=q0, true_running=False)
    nue_energies_testHyp, _, _, nue_wgts_testHyp = eves_gen_nue_testHyp.get_energy_theta_phi_wgt_arrays()
    numubar_energies_testHyp, _, _, numubar_wgts_testHyp = eves_gen_numubar_testHyp.get_energy_theta_phi_wgt_arrays()

    h_nuebar_test, _ = np.histogram(1e-3*sqrt(2*M_E*nuebar_energies_testHyp), weights=nuebar_wgts_testHyp, bins=q_bins)
    h_nue_test, _ = np.histogram(1e-3*sqrt(2*M_E*nue_energies_testHyp), weights=nue_wgts_testHyp, bins=q_bins)
    h_numubar_test, _ = np.histogram(1e-3*sqrt(2*M_E*numubar_energies_testHyp), weights=numubar_wgts_testHyp, bins=q_bins)
    h_numu_test, _ = np.histogram(1e-3*sqrt(2*M_E*numu_energies_testHyp), weights=numu_wgts_testHyp, bins=q_bins)

    # Build deltaChi^2's comparing test hypothesis to the null (with running)
    chi2_nuebar = np.sum((h_nuebar_test-h_nuebar_run)**2 / (h_nuebar_run + 1))
    chi2_numu = np.sum((h_numu_test-h_numu_run)**2 / (h_numu_run + 1))
    chi2_nue = np.sum((h_nue_test-h_nue_run)**2 / (h_nue_run + 1))
    chi2_numubar = np.sum((h_numubar_test-h_numubar_run)**2 / (h_numubar_run + 1))

    chi2_total = chi2_nuebar + chi2_numu + chi2_nue + chi2_numubar

    return -0.5 * chi2_total  # return negative log likelihood



def run_multinest_2param(muon_com_energy, scenario="MC", resume=False):
    # Run MultiNest

    pymultinest.run(loglike, prior_3TeV, 2, outputfiles_basename=f'multinest/weak_mixing_{muon_com_energy}/{muon_com_energy}_{scenario}_SSW2param2_',
                    n_live_points=2000, resume=resume, verbose=True)




def main():
    run_multinest_2param("3TeV", resume=False)



if __name__ == "__main__":
    main()
