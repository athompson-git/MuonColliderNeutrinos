import numpy as np

# Averaged ssw values and q values

fluxes_dict_muminus = {
    "500GeV": {
        "SSW": 0.23749462059310783,
        "Q": 0.18954159183676164
    },
    "3TeV": {
        "SSW": 0.2370024038686747,
        "Q": 0.46397084998088256
    },
    "10TeV": {
        "SSW": 0.2365412660420519,
        "Q": 0.8462992606999028
    }
}

fluxes_dict_muplus = {
    "500GeV": {
        "SSW": 0.23743605166405937,
        "Q": 0.21873393526869891
    },
    "3TeV": {
        "SSW": 0.23690402177488004,
        "Q": 0.5360353819918382
    },
    "10TeV": {
        "SSW": 0.23641179871496684,
        "Q": 0.9781049328857653
    }
}


# Q bins to guarantee enough stats (GeV)
q_bins_500 = np.logspace(np.log10(0.005), np.log10(0.55), 25)
q_centers_500 = (q_bins_500[1:] + q_bins_500[:-1])/2

q_bins_3TeV = np.logspace(np.log10(0.01), np.log10(1.4), 25)
q_centers_3TeV = (q_bins_3TeV[1:] + q_bins_3TeV[:-1])/2

q_bins_10TeV = np.logspace(-1.69, 0.34, 26)
q_centers_10TeV = (q_bins_10TeV[1:] + q_bins_10TeV[:-1])/2