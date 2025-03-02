import numpy as np
import pymultinest
from scipy.stats import chi2

from muon_collider_mc import *
import matplotlib.pyplot as plt
from matplotlib import cm, ticker
from matplotlib.colors import LogNorm

from flux_config import *

# SET CENTER OF MASS ENERGY AS INPUT
COM_ENERGY = "10TeV"
MUON_ENERGY = fluxes_dict[COM_ENERGY]["Emu"]  # GeV
ER_BINS = fluxes_dict[COM_ENERGY]["er_bins_MeV"]  # GeV

 # Define the event generators and null hypothesis spectrum
precalc_nue_flux = fluxes_dict[COM_ENERGY]["nue_flux"]
precalc_numu_flux = fluxes_dict[COM_ENERGY]["numu_flux"]

# Mu- decays
eves_gen_nuebar = EvESFromNeutrinoFlux(nu_flux=precalc_nue_flux, flavor="ebar", detector_material=Material("Ar"),
                                        detector_length=10.0, weak_mixing_angle_squared=fluxes_dict[COM_ENERGY]["SSW_MUM"],
                                        energy_only_flux=True)
eves_gen_numu = EvESFromNeutrinoFlux(nu_flux=precalc_numu_flux, flavor="mu", detector_material=Material("Ar"),
                                        detector_length=10.0, weak_mixing_angle_squared=fluxes_dict[COM_ENERGY]["SSW_MUM"],
                                        energy_only_flux=True)

# Mu+ decays
eves_gen_nue = EvESFromNeutrinoFlux(nu_flux=precalc_nue_flux, flavor="e", detector_material=Material("Ar"),
                                        detector_length=10.0, weak_mixing_angle_squared=fluxes_dict[COM_ENERGY]["SSW_MUP"],
                                        energy_only_flux=True)
eves_gen_numubar = EvESFromNeutrinoFlux(nu_flux=precalc_numu_flux, flavor="mubar", detector_material=Material("Ar"),
                                        detector_length=10.0, weak_mixing_angle_squared=fluxes_dict[COM_ENERGY]["SSW_MUP"],
                                        energy_only_flux=True)

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
def loglike_sr_2param_minus(cube, ndim, nparams):
    g_nue = cube[0]
    g_numu = cube[1]

    gLe_mod = 2*g_nue
    gLmu_mod = 2*g_numu
    gRe_mod = 2*g_nue
    gRmu_mod = 2*g_numu

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

def loglike_sr_2param_plus(cube, ndim, nparams):
    g_nue = cube[0]
    g_numu = cube[1]

    gLe_mod = 2*g_nue
    gLmu_mod = 2*g_numu
    gRe_mod = 2*g_nue
    gRmu_mod = 2*g_numu

    # Generate events based on parameters (assuming predefined fluxes and material)
    eves_gen_nue.simulate_eves_rectangle_rule(n_samples=100, gL_mod=gLe_mod, gR_mod=gRe_mod)
    eves_gen_numubar.simulate_eves_rectangle_rule(n_samples=100, gL_mod=gLmu_mod, gR_mod=gRmu_mod)

    nue_e, _, _, nue_wgts = eves_gen_nue.get_energy_theta_phi_wgt_arrays()
    numubar_e, _, _, numubar_wgts = eves_gen_numubar.get_energy_theta_phi_wgt_arrays()

    signal_spectrum = np.histogram(1e-3*nue_e, weights=nue_wgts, bins=ER_BINS)[0] + \
                      np.histogram(1e-3*numubar_e, weights=numubar_wgts, bins=ER_BINS)[0]

    chi2_val = np.sum((signal_spectrum - null_hypothesis_spectrum_muminus)**2 / null_hypothesis_spectrum_muminus)

    # Return the log-likelihood
    return -0.5 * chi2_val

# Define the log-likelihood function: muon collider mu+ mu-
def loglike_mc_2param(cube, ndim, nparams):
    g_nue = cube[0]
    g_numu = cube[1]

    gLe_mod = 2*g_nue
    gLmu_mod = 2*g_numu
    gRe_mod = 2*g_nue
    gRmu_mod = 2*g_numu

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


def prior_2param(cube, ndim, nparams):
    cube[0] = cube[0]  # g_nue  ~ [0, 1]
    cube[1] = cube[1]  # g_numu ~ [0, 1]

# Setup the analysis

def run_multinest_2param(muon_com_energy, scenario, resume=False):
    # Run MultiNest
    
    if scenario == "MC":
        pymultinest.run(loglike_mc_2param, prior_2param, 2, outputfiles_basename=f'multinest/{scenario}/{muon_com_energy}_{scenario}_nuflavor_',
                        n_live_points=500, resume=resume, verbose=True)
    elif scenario == "SR_MINUS":
        pymultinest.run(loglike_sr_2param_minus, prior_2param, 2, outputfiles_basename=f'multinest/{scenario}/{muon_com_energy}_{scenario}_nuflavor_',
                        n_live_points=500, resume=resume, verbose=True)
    elif scenario == "SR_PLUS":
        pymultinest.run(loglike_sr_2param_plus, prior_2param, 2, outputfiles_basename=f'multinest/{scenario}/{muon_com_energy}_{scenario}_nuflavor_',
                        n_live_points=500, resume=resume, verbose=True)



if __name__ == "__main__":
    # Define your parameters for the run
    
    # COM energies: 500 GeV, 3 TeV, 10 TeV

    # 2 scenarios: Muon Collider (MC) and Storage Ring (SR)

    run_multinest_2param("sqrts-10TeV", "MC", resume=False)
