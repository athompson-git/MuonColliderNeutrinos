import numpy as np
from muon_collider_mc import *



# Load fluxes: energy spectra
numu_flux_500GeV = np.genfromtxt("data/numu_energy_flux_MeV_SqrtS-500GeV_1e19nu.txt")
nue_flux_500GeV = np.genfromtxt("data/nuebar_energy_flux_MeV_SqrtS-500GeV_1e19nu.txt")

numu_flux_3TeV = np.genfromtxt("data/numu_energy_flux_MeV_SqrtS-3TeV_1e19nu.txt")
nue_flux_3TeV = np.genfromtxt("data/nuebar_energy_flux_MeV_SqrtS-3TeV_1e19nu.txt")

numu_flux_10TeV = np.genfromtxt("data/numu_energy_flux_MeV_SqrtS-10TeV_1e19nu.txt")
nue_flux_10TeV = np.genfromtxt("data/nuebar_energy_flux_MeV_SqrtS-10TeV_1e19nu.txt")


# Q bins to guarantee enough stats (GeV)
q_bins_500 = np.logspace(np.log10(0.005), np.log10(0.55), 25)
er_bins_GeV_500 = (q_bins_500)**2 / (2*M_E*1e-3)
q_centers_500 = (q_bins_500[1:] + q_bins_500[:-1])/2

q_bins_3TeV = np.logspace(np.log10(0.01), np.log10(1.4), 25)
er_bins_GeV_3TeV = (1e3*q_bins_3TeV)**2 / (2*M_E*1e-3)
q_centers_3TeV = (q_bins_3TeV[1:] + q_bins_3TeV[:-1])/2

q_bins_10TeV = np.logspace(-1.69, 0.34, 26)
er_bins_GeV_10TeV = (q_bins_10TeV)**2 / (2*M_E*1e-3)
q_centers_10TeV = (q_bins_10TeV[1:] + q_bins_10TeV[:-1])/2


# Averaged ssw values and q values
fluxes_dict = {
    "500GeV": {
        "numu_flux": numu_flux_500GeV,
        "nue_flux": nue_flux_500GeV,
        "Emu": 250.0,
        "SSW_MUM": 0.23749462059310783,
        "Q_MUM": 0.18954159183676164,
        "SSW_MUP": 0.23743605166405937,
        "Q_MUP": 0.21873393526869891,
        "er_bins_GeV": er_bins_GeV_500,
        "q_bins_GeV": q_bins_500
    },
    "3TeV": {
        "numu_flux": numu_flux_3TeV,
        "nue_flux": nue_flux_3TeV,
        "Emu": 1500.0,
        "SSW_MUM": 0.2370024038686747,
        "Q_MUM": 0.46397084998088256,
        "SSW_MUP": 0.23690402177488004,
        "Q_MUP": 0.5360353819918382,
        "er_bins_GeV": er_bins_GeV_3TeV,
        "q_bins_GeV": q_bins_3TeV
    },
    "10TeV": {
        "numu_flux": numu_flux_10TeV,
        "nue_flux": nue_flux_10TeV,
        "Emu": 5000.0,
        "SSW_MUM": 0.2365412660420519,
        "Q_MUM": 0.8462992606999028,
        "SSW_MUP": 0.23641179871496684,
        "Q_MUP": 0.9781049328857653,
        "er_bins_GeV": er_bins_GeV_10TeV,
        "q_bins_GeV": q_bins_10TeV
    }
}
