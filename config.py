import numpy as np

# Averaged ssw values and q values
fluxes_dict_muminus = {
    "500GeV": {
        "SSW": 0.2373526248780235,
        "Q": 0.262293422006969
    },
    "3TeV": {
        "SSW": 0.23697467683471973,
        "Q": 0.4824816351073696
    },
    "10TeV": {
        "SSW": 0.23650875776108485,
        "Q": 0.879394682695072
    }
}

fluxes_dict_muplus = {
    "500GeV": {
        "SSW": 0.23729171497873042,
        "Q": 0.2957663115564079
    },
    "3TeV": {
        "SSW": 0.23690081638027105,
        "Q": 0.5384649446012822
    },
    "10TeV": {
        "SSW": 0.2364087075207415,
        "Q": 0.9812519580225515
    }
}


# Q bins to guarantee enough stats (GeV)
q_bins_500 = np.logspace(np.log10(0.005), np.log10(0.55), 25)
q_centers_500 = (q_bins_500[1:] + q_bins_500[:-1])/2

q_bins_3TeV = np.logspace(np.log10(0.01), np.log10(1.4), 25)
q_centers_3TeV = (q_bins_3TeV[1:] + q_bins_3TeV[:-1])/2

q_bins_10TeV = np.logspace(-1.69, 0.34, 26)
q_centers_10TeV = (q_bins_10TeV[1:] + q_bins_10TeV[:-1])/2