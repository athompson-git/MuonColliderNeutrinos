import numpy as np
import pymultinest
from scipy.stats import chi2

from muon_collider_mc import *
import matplotlib.pyplot as plt
from matplotlib import cm, ticker
from matplotlib.colors import LogNorm


# Load the precalculated fluxes
numu_flux_500GeV = np.genfromtxt("data/numu_flux_MeV_rad_SqrtS-500GeV_216e19Mu_25m2_by_10m_200m.txt")
nue_flux_500GeV = np.genfromtxt("data/nuebar_flux_MeV_rad_SqrtS-500GeV_216e19Mu_25m2_by_10m_200m.txt")

numu_flux_3TeV = np.genfromtxt("data/numu_flux_MeV_rad_SqrtS-3TeV_216e19Mu_25m2_by_10m_200m.txt")
nue_flux_3TeV = np.genfromtxt("data/nuebar_flux_MeV_rad_SqrtS-3TeV_216e19Mu_25m2_by_10m_200m.txt")

numu_flux_10TeV = np.genfromtxt("data/numu_flux_MeV_rad_SqrtS-10TeV_216e19Mu_25m2_by_10m_200m.txt")
nue_flux_10TeV = np.genfromtxt("data/nuebar_flux_MeV_rad_SqrtS-10TeV_216e19Mu_25m2_by_10m_200m.txt")



fluxes_dict = {
    "500GeV": {
        "numu_flux": numu_flux_500GeV,
        "nue_flux": nue_flux_500GeV,
        "SSW": 0.23739130434782607  # at 0.24 GeV
    },
    "3TeV": {
        "numu_flux": numu_flux_3TeV,
        "nue_flux": nue_flux_3TeV,
        "SSW": 0.23667701863354035  # at 0.7 GeV
    },
    "10TeV": {
        "numu_flux": numu_flux_10TeV,
        "nue_flux": nue_flux_10TeV,
        "SSW": 0.23596273291925465  # at 1.5 GeV
    }
}

COM_ENERGY = "3TeV"
MUON_ENERGY = 1500.0  # GeV
ER_BINS = np.logspace(-1, np.log10(MUON_ENERGY), 50)  # GeV

 # Define the event generators and null hypothesis spectrum
precalc_nue_flux = fluxes_dict[COM_ENERGY]["nue_flux"]
precalc_numu_flux = fluxes_dict[COM_ENERGY]["numu_flux"]

# Mu- decays
eves_gen_nuebar = EvESFromNeutrinoFlux(nu_flux=precalc_nue_flux, flavor="ebar", detector_material=Material("Ar"),
                                        detector_length=10.0, weak_mixing_angle_squared=fluxes_dict[COM_ENERGY]["SSW"])
eves_gen_numu = EvESFromNeutrinoFlux(nu_flux=precalc_numu_flux, flavor="mu", detector_material=Material("Ar"),
                                        detector_length=10.0, weak_mixing_angle_squared=fluxes_dict[COM_ENERGY]["SSW"])

# Mu+ decays
eves_gen_nue = EvESFromNeutrinoFlux(nu_flux=precalc_nue_flux, flavor="e", detector_material=Material("Ar"),
                                        detector_length=10.0, weak_mixing_angle_squared=fluxes_dict[COM_ENERGY]["SSW"])
eves_gen_numubar = EvESFromNeutrinoFlux(nu_flux=precalc_numu_flux, flavor="mubar", detector_material=Material("Ar"),
                                        detector_length=10.0, weak_mixing_angle_squared=fluxes_dict[COM_ENERGY]["SSW"])

eves_gen_nuebar.simulate_eves_rectangle_rule(n_samples=100)
eves_gen_numu.simulate_eves_rectangle_rule(n_samples=100)
eves_gen_nue.simulate_eves_rectangle_rule(n_samples=100)
eves_gen_numubar.simulate_eves_rectangle_rule(n_samples=100)

nuebar_energies, _, _, nuebar_wgts = eves_gen_nuebar.get_energy_theta_phi_wgt_arrays()
numu_energies, _, _, numu_wgts = eves_gen_numu.get_energy_theta_phi_wgt_arrays()
nue_energies, _, _, nue_wgts = eves_gen_nue.get_energy_theta_phi_wgt_arrays()
numubar_energies, _, _, numubar_wgts = eves_gen_numubar.get_energy_theta_phi_wgt_arrays()

# Null hypothesis spectrum
nuebar_spectrum_null = np.histogram(1e-3*nuebar_energies, weights=nuebar_wgts, bins=ER_BINS)[0] 
numu_spectrum_null = np.histogram(1e-3*numu_energies, weights=numu_wgts, bins=ER_BINS)[0]
nue_spectrum_null = np.histogram(1e-3*nue_energies, weights=nue_wgts, bins=ER_BINS)[0] 
numubar_spectrum_null = np.histogram(1e-3*numubar_energies, weights=numubar_wgts, bins=ER_BINS)[0]

null_hypothesis_spectrum_muminus = nuebar_spectrum_null + numu_spectrum_null
null_hypothesis_spectrum_muplus = nue_spectrum_null + numubar_spectrum_null

# check that no bin has too few events
print("Minimum expected events per bin in mu- = {}, mu+ = {}".format(min(null_hypothesis_spectrum_muminus),
                                                                     min(null_hypothesis_spectrum_muplus)))


# Define the log-likelihood function: storage ring
def loglike_sr(cube, ndim, nparams):
    g_nue = cube[0]
    g_numu = cube[1]
    g_L = cube[2]
    g_R = cube[3]

    # TODO: divide by SM values
    gL_SM = (fluxes_dict[COM_ENERGY]["SSW"] - 0.5)
    gR_SM = fluxes_dict[COM_ENERGY]["SSW"]

    gLe_mod = 2*g_nue*g_L / gL_SM
    gLmu_mod = 2*g_numu*g_L / gL_SM
    gRe_mod = 2*g_nue*g_R / gR_SM
    gRmu_mod = 2*g_numu*g_R / gR_SM

    # Generate events based on parameters (assuming predefined fluxes and material)
    eves_gen_nuebar.simulate_eves_rectangle_rule(n_samples=100, gL_mod=gLe_mod, gR_mod=gRe_mod)
    eves_gen_numu.simulate_eves_rectangle_rule(n_samples=100, gL_mod=gLmu_mod, gR_mod=gRmu_mod)

    nuebar_e, _, _, nuebar_wgts = eves_gen_nuebar.get_energy_theta_phi_wgt_arrays()
    numu_e, _, _, numu_wgts = eves_gen_numu.get_energy_theta_phi_wgt_arrays()

    signal_spectrum = np.histogram(1e-3*nuebar_e, weights=nuebar_wgts, bins=ER_BINS)[0] + \
                      np.histogram(1e-3*numu_e, weights=numu_wgts, bins=ER_BINS)[0]

    chi2_val = np.sum((signal_spectrum - null_hypothesis_spectrum_muminus)**2 / null_hypothesis_spectrum_muminus)

    # Return the log-likelihood
    return -0.5 * chi2_val

# Define the log-likelihood function: muon collider mu+ mu-
def loglike_mc(cube, ndim, nparams):
    g_nue = cube[0]
    g_numu = cube[1]
    g_L = cube[2]
    g_R = cube[3]

    gL_SM = (fluxes_dict[COM_ENERGY]["SSW"] - 0.5)
    gR_SM = fluxes_dict[COM_ENERGY]["SSW"]

    gLe_mod = 2*g_nue*g_L / gL_SM
    gLmu_mod = 2*g_numu*g_L / gL_SM
    gRe_mod = 2*g_nue*g_R / gR_SM
    gRmu_mod = 2*g_numu*g_R / gR_SM

    # Generate events based on parameters (assuming predefined fluxes and material)
    eves_gen_nuebar.simulate_eves_rectangle_rule(n_samples=100, gL_mod=gLe_mod, gR_mod=gRe_mod)
    eves_gen_numu.simulate_eves_rectangle_rule(n_samples=100, gL_mod=gLmu_mod, gR_mod=gRmu_mod)

    eves_gen_nue.simulate_eves_rectangle_rule(n_samples=100, gL_mod=gLe_mod, gR_mod=gRe_mod)
    eves_gen_numubar.simulate_eves_rectangle_rule(n_samples=100, gL_mod=gLmu_mod, gR_mod=gRmu_mod)

    nuebar_e, _, _, nuebar_wgts = eves_gen_nuebar.get_energy_theta_phi_wgt_arrays()
    numu_e, _, _, numu_wgts = eves_gen_numu.get_energy_theta_phi_wgt_arrays()

    nue_e, _, _, nue_wgts = eves_gen_nue.get_energy_theta_phi_wgt_arrays()
    numubar_e, _, _, numubar_wgts = eves_gen_numubar.get_energy_theta_phi_wgt_arrays()

    signal_spectrum_muminus = np.histogram(1e-3*nuebar_e, weights=nuebar_wgts, bins=ER_BINS)[0] + \
                              np.histogram(1e-3*numu_e, weights=numu_wgts, bins=ER_BINS)[0]
    
    signal_spectrum_muplus = np.histogram(1e-3*nue_e, weights=nue_wgts, bins=ER_BINS)[0] + \
                              np.histogram(1e-3*numubar_e, weights=numubar_wgts, bins=ER_BINS)[0]

    chi2_muminus = np.sum((signal_spectrum_muminus - null_hypothesis_spectrum_muminus)**2 / null_hypothesis_spectrum_muminus)
    chi2_muplus = np.sum((signal_spectrum_muplus - null_hypothesis_spectrum_muplus)**2 / null_hypothesis_spectrum_muplus)

    # Return the log-likelihood
    return -0.5 * (chi2_muplus + chi2_muminus)


# Define the log-likelihood function: storage ring
def loglike_sr_2param(cube, ndim, nparams):
    g_L = cube[0]
    g_R = cube[1]

    # Generate events based on parameters (assuming predefined fluxes and material)
    eves_gen_nuebar.simulate_eves_rectangle_rule(n_samples=100, gL_mod=g_L, gR_mod=g_R)
    eves_gen_numu.simulate_eves_rectangle_rule(n_samples=100, gL_mod=g_L, gR_mod=g_R)

    nuebar_e, _, _, nuebar_wgts = eves_gen_nuebar.get_energy_theta_phi_wgt_arrays()
    numu_e, _, _, numu_wgts = eves_gen_numu.get_energy_theta_phi_wgt_arrays()

    signal_spectrum = np.histogram(1e-3*nuebar_e, weights=nuebar_wgts, bins=ER_BINS)[0] + \
                      np.histogram(1e-3*numu_e, weights=numu_wgts, bins=ER_BINS)[0]

    chi2_val = np.sum((signal_spectrum - null_hypothesis_spectrum_muminus)**2 / null_hypothesis_spectrum_muminus)

    # Return the log-likelihood
    return -0.5 * chi2_val

# Define the log-likelihood function: muon collider mu+ mu-
def loglike_mc_2param(cube, ndim, nparams):
    g_L = cube[0]
    g_R = cube[1]

    # Generate events based on parameters (assuming predefined fluxes and material)
    eves_gen_nuebar.simulate_eves_rectangle_rule(n_samples=100, gL_mod=g_L, gR_mod=g_R)
    eves_gen_numu.simulate_eves_rectangle_rule(n_samples=100, gL_mod=g_L, gR_mod=g_R)

    eves_gen_nue.simulate_eves_rectangle_rule(n_samples=100, gL_mod=g_L, gR_mod=g_R)
    eves_gen_numubar.simulate_eves_rectangle_rule(n_samples=100, gL_mod=g_L, gR_mod=g_R)

    nuebar_e, _, _, nuebar_wgts = eves_gen_nuebar.get_energy_theta_phi_wgt_arrays()
    numu_e, _, _, numu_wgts = eves_gen_numu.get_energy_theta_phi_wgt_arrays()

    nue_e, _, _, nue_wgts = eves_gen_nue.get_energy_theta_phi_wgt_arrays()
    numubar_e, _, _, numubar_wgts = eves_gen_numubar.get_energy_theta_phi_wgt_arrays()

    signal_spectrum_muminus = np.histogram(1e-3*nuebar_e, weights=nuebar_wgts, bins=ER_BINS)[0] + \
                              np.histogram(1e-3*numu_e, weights=numu_wgts, bins=ER_BINS)[0]
    
    signal_spectrum_muplus = np.histogram(1e-3*nue_e, weights=nue_wgts, bins=ER_BINS)[0] + \
                              np.histogram(1e-3*numubar_e, weights=numubar_wgts, bins=ER_BINS)[0]

    chi2_muminus = np.sum((signal_spectrum_muminus - null_hypothesis_spectrum_muminus)**2 / null_hypothesis_spectrum_muminus)
    chi2_muplus = np.sum((signal_spectrum_muplus - null_hypothesis_spectrum_muplus)**2 / null_hypothesis_spectrum_muplus)

    # Return the log-likelihood
    return -0.5 * (chi2_muplus + chi2_muminus)

# Define the prior function
def prior(cube, ndim, nparams):
    # Define flat priors
    cube[0] = cube[0]  # g_nue, [0, 1]
    cube[1] = cube[1]  # g_numu, [0, 1]
    cube[2] = cube[2] * 0.6 - 0.3  # g_L, [-0.3, 0.3]
    cube[3] = cube[3] * 0.6 - 0.3  # g_R, [-0.3, 0.3]

def prior_2param(cube, ndim, nparams):
    cube[0] = cube[0] * 0.6 - 0.3  # g_L, [-0.3, 0.3]
    cube[1] = cube[1] * 0.6 - 0.3  # g_R, [-0.3, 0.3]

# Setup the analysis
def run_multinest(muon_com_energy, scenario, resume=False):
    # Run MultiNest
    
    if scenario == "MC":
        pymultinest.run(loglike_mc, prior, 4, outputfiles_basename=f'multinest/{scenario}/{muon_com_energy}_{scenario}_',
                        n_live_points=500, resume=resume, verbose=True)
    elif scenario == "SR":
        pymultinest.run(loglike_sr, prior, 4, outputfiles_basename=f'multinest/{scenario}/{muon_com_energy}_{scenario}_',
                        n_live_points=500, resume=resume, verbose=True)


def run_multinest_2param(muon_com_energy, scenario, resume=False):
    # Run MultiNest
    
    if scenario == "MC":
        pymultinest.run(loglike_mc_2param, prior_2param, 2, outputfiles_basename=f'multinest/{scenario}/{muon_com_energy}_{scenario}_2param_',
                        n_live_points=500, resume=resume, verbose=True)
    elif scenario == "SR":
        pymultinest.run(loglike_sr_2param, prior_2param, 2, outputfiles_basename=f'multinest/{scenario}/{muon_com_energy}_{scenario}_2param_',
                        n_live_points=500, resume=resume, verbose=True)



if __name__ == "__main__":
    # Define your parameters for the run
    
    # COM energies: 500 GeV, 3 TeV, 10 TeV

    # 2 scenarios: Muon Collider (MC) and Storage Ring (SR)

    run_multinest("sqrts-3TeV", "MC", resume=False)
