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

def log_pois(data, bkg, signal):
    return -2*np.sum(data * np.log(signal + bkg) - (signal + bkg) - gammaln(data + 1))

# Define energy range, samples, parameters etc.
N_SAMPLES = 500

q_bins = np.logspace(-1.69, 0.34, 26)

q0_10TeV = 1500.0  # at 1.5 GeV for sqrt(s) = 10 TeV

fluxes_dict = {
    "500GeV": {
        "SSW": 0.23739130434782607  # at 0.24 GeV
    },
    "3TeV": {
        "SSW": 0.23667701863354035  # at 0.7 GeV
    },
    "10TeV": {
        "SSW": 0.23596273291925465  # at 1.5 GeV
    }
}

# Read in fluxes
numu_flux_10TeV = np.genfromtxt("data/numu_flux_MeV_rad_SqrtS-10TeV_216e19Mu_25m2_by_10m_200m.txt")
nue_flux_10TeV = np.genfromtxt("data/nuebar_flux_MeV_rad_SqrtS-10TeV_216e19Mu_25m2_by_10m_200m.txt")



# Generate NULL HYPOTHESIS
# sqrt(s) = 10 TeV - WITH RUNNING
eves_gen_nuebar_10TeV_running = EvESFromNeutrinoFlux(nu_flux=nue_flux_10TeV, flavor="ebar", detector_material=Material("Ar"),
                                    detector_length=10.0, ssw_running=True)
eves_gen_numu_10TeV_running = EvESFromNeutrinoFlux(nu_flux=numu_flux_10TeV, flavor="mu", detector_material=Material("Ar"),
                                    detector_length=10.0, ssw_running=True)
eves_gen_nuebar_10TeV_running.simulate_eves_rectangle_rule(n_samples=N_SAMPLES)
eves_gen_numu_10TeV_running.simulate_eves_rectangle_rule(n_samples=N_SAMPLES)
nuebar_energies_10TeV_running, _, _, nuebar_wgts_10TeV_running = eves_gen_nuebar_10TeV_running.get_energy_theta_phi_wgt_arrays()
numu_energies_10TeV_running, _, _, numu_wgts_10TeV_running = eves_gen_numu_10TeV_running.get_energy_theta_phi_wgt_arrays()

eves_gen_nue_10TeV_running = EvESFromNeutrinoFlux(nu_flux=nue_flux_10TeV, flavor="e", detector_material=Material("Ar"),
                                    detector_length=10.0, ssw_running=True)
eves_gen_numubar_10TeV_running = EvESFromNeutrinoFlux(nu_flux=numu_flux_10TeV, flavor="mubar", detector_material=Material("Ar"),
                                    detector_length=10.0, ssw_running=True)
eves_gen_nue_10TeV_running.simulate_eves_rectangle_rule(n_samples=N_SAMPLES)
eves_gen_numubar_10TeV_running.simulate_eves_rectangle_rule(n_samples=N_SAMPLES)
nue_energies_10TeV_running, _, _, nue_wgts_10TeV_running = eves_gen_nue_10TeV_running.get_energy_theta_phi_wgt_arrays()
numubar_energies_10TeV_running, _, _, numubar_wgts_10TeV_running = eves_gen_numubar_10TeV_running.get_energy_theta_phi_wgt_arrays()


h_nuebar_run, _ = np.histogram(1e-3*sqrt(2*M_E*nuebar_energies_10TeV_running), weights=nuebar_wgts_10TeV_running, bins=q_bins)
h_nue_run, _ = np.histogram(1e-3*sqrt(2*M_E*nue_energies_10TeV_running), weights=nue_wgts_10TeV_running, bins=q_bins)
h_numubar_run, _ = np.histogram(1e-3*sqrt(2*M_E*numubar_energies_10TeV_running), weights=numubar_wgts_10TeV_running, bins=q_bins)
h_numu_run, _ = np.histogram(1e-3*sqrt(2*M_E*numu_energies_10TeV_running), weights=numu_wgts_10TeV_running, bins=q_bins)


print("Setting up null hypothesis predictions...")
print("h_nuebar events = {}".format(h_nuebar_run))
print("h_numu events = {}".format(h_numu_run))
print("h_numubar events = {}".format(h_numubar_run))
print("h_nue events = {}".format(h_nue_run))


# SET UP TEST HYPOTHESIS GENERATORS
eves_gen_nuebar_10TeV_testHyp = EvESFromNeutrinoFluxRunningSSW(nu_flux=nue_flux_10TeV, flavor="ebar", detector_material=Material("Ar"),
                                    detector_length=10.0)
eves_gen_numu_10TeV_testHyp = EvESFromNeutrinoFluxRunningSSW(nu_flux=numu_flux_10TeV, flavor="mu", detector_material=Material("Ar"),
                                    detector_length=10.0)


eves_gen_nue_10TeV_testHyp = EvESFromNeutrinoFluxRunningSSW(nu_flux=nue_flux_10TeV, flavor="e", detector_material=Material("Ar"),
                                    detector_length=10.0)
eves_gen_numubar_10TeV_testHyp = EvESFromNeutrinoFluxRunningSSW(nu_flux=numu_flux_10TeV, flavor="mubar", detector_material=Material("Ar"),
                                    detector_length=10.0)




def prior(cube, ndim, nparams):
    cube[0] = 0.2345 + 0.003 * cube[0]  # SW^2 (0) constant piece
    cube[1] = 10**(4.0*cube[1] - 6.0)  # dSW^2/dq (1) linear piece


def loglike(cube, ndim, nparams):
    a = cube[0]
    b = cube[1]
    eves_gen_nuebar_10TeV_testHyp.simulate_eves_rectangle_rule(n_samples=N_SAMPLES, sw20=a, sw2_deriv=b, q0=q0_10TeV, true_running=False)
    eves_gen_numu_10TeV_testHyp.simulate_eves_rectangle_rule(n_samples=N_SAMPLES, sw20=a, sw2_deriv=b, q0=q0_10TeV, true_running=False)
    nuebar_energies_10TeV_testHyp, _, _, nuebar_wgts_10TeV_testHyp = eves_gen_nuebar_10TeV_testHyp.get_energy_theta_phi_wgt_arrays()
    numu_energies_10TeV_testHyp, _, _, numu_wgts_10TeV_testHyp = eves_gen_numu_10TeV_testHyp.get_energy_theta_phi_wgt_arrays()

    eves_gen_nue_10TeV_testHyp.simulate_eves_rectangle_rule(n_samples=N_SAMPLES, sw20=a, sw2_deriv=b, q0=q0_10TeV, true_running=False)
    eves_gen_numubar_10TeV_testHyp.simulate_eves_rectangle_rule(n_samples=N_SAMPLES, sw20=a, sw2_deriv=b, q0=q0_10TeV, true_running=False)
    nue_energies_10TeV_testHyp, _, _, nue_wgts_10TeV_testHyp = eves_gen_nue_10TeV_testHyp.get_energy_theta_phi_wgt_arrays()
    numubar_energies_10TeV_testHyp, _, _, numubar_wgts_10TeV_testHyp = eves_gen_numubar_10TeV_testHyp.get_energy_theta_phi_wgt_arrays()

    h_nuebar_test, _ = np.histogram(1e-3*sqrt(2*M_E*nuebar_energies_10TeV_testHyp), weights=nuebar_wgts_10TeV_testHyp, bins=q_bins)
    h_nue_test, _ = np.histogram(1e-3*sqrt(2*M_E*nue_energies_10TeV_testHyp), weights=nue_wgts_10TeV_testHyp, bins=q_bins)
    h_numubar_test, _ = np.histogram(1e-3*sqrt(2*M_E*numubar_energies_10TeV_testHyp), weights=numubar_wgts_10TeV_testHyp, bins=q_bins)
    h_numu_test, _ = np.histogram(1e-3*sqrt(2*M_E*numu_energies_10TeV_testHyp), weights=numu_wgts_10TeV_testHyp, bins=q_bins)

    # Build deltaChi^2's comparing test hypothesis to the null (with running)
    chi2_nuebar = np.sum((h_nuebar_test-h_nuebar_run)**2 / (h_nuebar_run + 1))
    chi2_numu = np.sum((h_numu_test-h_numu_run)**2 / (h_numu_run + 1))
    chi2_nue = np.sum((h_nue_test-h_nue_run)**2 / (h_nue_run + 1))
    chi2_numubar = np.sum((h_numubar_test-h_numubar_run)**2 / (h_numubar_run + 1))

    chi2_total = chi2_nuebar + chi2_numu + chi2_nue + chi2_numubar

    return -0.5 * chi2_total  # return negative log likelihood



def run_multinest_2param(muon_com_energy, scenario="MC", resume=False):
    # Run MultiNest

    pymultinest.run(loglike, prior, 2, outputfiles_basename=f'multinest/weak_mixing/{muon_com_energy}_{scenario}_SSW2param_',
                    n_live_points=500, resume=resume, verbose=True)




def main():
    run_multinest_2param("10TeV")



if __name__ == "__main__":
    main()