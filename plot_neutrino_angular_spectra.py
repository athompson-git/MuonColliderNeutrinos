from muon_decay import *

import matplotlib.pyplot as plt
from matplotlib import cm, ticker
from matplotlib.pylab import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

Emu = 5.0e6  # 5 TeV

l_det = 100.0  # meters

Enus = np.linspace(0.0, 1.0, 15)*Emu

thetas = np.logspace(-6, -2, 1000)

colors = plt.cm.inferno(np.linspace(0.0, 1.0, len(Enus)))




for i, enu in enumerate(Enus):
    d2n_nu_e = d2NdydOmega_nue(enu/Emu, thetas, Emu, l_det, 1.0)
    d2n_nu_mu = d2NdydOmega_numu(enu/Emu, thetas, Emu, l_det, 1.0)

    plt.plot(thetas, d2n_nu_e, ls='solid', color=colors[i])
    #plt.plot(thetas, d2n_nu_mu, ls='solid', color=colors[i])

sm = plt.cm.ScalarMappable(cmap="inferno")


cbar = plt.colorbar(sm)
cbar.set_label(label=r"$y$", weight='bold', fontsize=14)

plt.xscale('log')
plt.yscale('log')
plt.xlim((thetas[0], thetas[-1]))
plt.ylabel(r"$d^2 N_e / (dy d\Omega)$ [sr$^{-1}$ m$^{-2}$ muon$^{-1}$]", fontsize=14)
plt.xlabel(r"$\theta_\nu$ [rad]", fontsize=14)
plt.title(r"$E_\mu = 5$ TeV, $L = 100$ m", loc="right", fontsize=14)
plt.ylim(bottom=5e-1)
plt.tight_layout()
plt.show()


# plot energy fraction dist.
thetas = np.logspace(-6, -2, 15)
colors = plt.cm.inferno(np.linspace(0.0, 1.0, len(thetas)))
ys = np.linspace(0.0, 1.0, 10000)
for i, th in enumerate(thetas):
    d2n_nu_e = d2NdydOmega_nue(ys, th, Emu, l_det, 1.0)
    d2n_nu_mu = d2NdydOmega_numu(ys, th, Emu, l_det, 1.0)

    plt.plot(ys, d2n_nu_e, ls='solid', color=colors[i])
    #plt.plot(thetas, d2n_nu_mu, ls='solid', color=colors[i])

sm = plt.cm.ScalarMappable(cmap="inferno")


cbar = plt.colorbar(sm)
cbar.set_label(label=r"$\theta$", weight='bold', fontsize=14)
plt.xscale('log')
plt.yscale('log')
plt.xlim((ys[0], ys[-1]))
plt.ylabel(r"$d^2 N_e / (dy d\Omega)$ [sr$^{-1}$ m$^{-2}$ muon$^{-1}$]", fontsize=14)
plt.xlabel(r"$y = E_\nu / E_\mu$", fontsize=14)
plt.title(r"$E_\mu = 5$ TeV, $L = 100$ m", loc="right", fontsize=14)
plt.ylim(bottom=5e-1)
plt.tight_layout()
plt.show()