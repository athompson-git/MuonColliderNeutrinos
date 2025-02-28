from muon_collider_mc import *
import matplotlib.pyplot as plt
from matplotlib import cm, ticker
from matplotlib.colors import LogNorm
from matplotlib.pylab import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

event_gen = NeutrinoFluxMuonRing(Emu=5.0e6, ring_radius=1000.0, N_muons=2.16e20, det_dist=200.0,
                                det_area=25.0, det_length=10.0, det_mat=Material("Ar"))


event_gen.simulate_flux(n_samples=50000000)



# create histogram
enu_bins = np.logspace(3, np.log10(event_gen.Emu), 100)
enu_centers = (enu_bins[1:] + enu_bins[:-1])/2
theta_bins = np.logspace(-7, -3, 100)
theta_bin_centers = (theta_bins[1:] + theta_bins[:-1])/2

h_numu, xedges_numu, yedges_numu = np.histogram2d(event_gen.flux_energies, event_gen.flux_thetas, weights=event_gen.flux_weights_numu,
                                                  bins=[enu_bins, theta_bins])

h_nue, xedges_nue, yedges_nue = np.histogram2d(event_gen.flux_energies, event_gen.flux_thetas, weights=event_gen.flux_weights_nue,
                                                  bins=[enu_bins, theta_bins])



plt.hist2d(event_gen.flux_energies, event_gen.flux_thetas, weights=event_gen.flux_weights_numu,
                                                  bins=[enu_bins, theta_bins])
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r"$E_{\nu_\mu}$ [MeV]", fontsize=16)
plt.ylabel(r"$\theta_{\nu_\mu}$ [rad]", fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.colorbar(label="Counts / 2.16e20 Muons")
plt.tight_layout()
plt.show()



plt.hist2d(event_gen.flux_energies, event_gen.flux_thetas, weights=event_gen.flux_weights_nue,
                                                  bins=[enu_bins, theta_bins])
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$E_{\nu_e}$ [MeV]", fontsize=16)
plt.ylabel(r"$\theta_{\nu_e}$ [rad]", fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.colorbar(label="Counts / 2.16e20 Muons")
plt.tight_layout()
plt.show()



# output to file
flattened_h_mu = h_numu.flatten()
flattened_h_e = h_nue.flatten()
flattened_energy_bins = np.repeat(enu_centers, 99)
flattened_theta_bins = np.tile(theta_bin_centers, 99)
print(flattened_h_e.shape, flattened_energy_bins.shape, flattened_theta_bins.shape)

output_array_numu = np.array([flattened_energy_bins, flattened_theta_bins, flattened_h_mu]).transpose()
output_array_nue = np.array([flattened_energy_bins, flattened_theta_bins, flattened_h_e]).transpose()
print(output_array_nue.shape)

# delete parts of the array less than 1e15 per bin (based on by eye)
trimmed_output_numu_array = output_array_numu[output_array_numu[:,2] >= 1e14]
trimmed_output_nue_array = output_array_nue[output_array_nue[:,2] >= 1e14]


# test that they dont hide too much info
my_cmap = cm.viridis
my_cmap.set_under('w',1)
plt.hist2d(trimmed_output_numu_array[:,0], trimmed_output_numu_array[:,1], weights=trimmed_output_numu_array[:,2],
                                                  bins=[enu_bins, theta_bins], cmap=my_cmap, norm=LogNorm())
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r"$E_{\nu_\mu}$ [MeV]", fontsize=16)
plt.ylabel(r"$\theta_{\nu_\mu}$ [rad]", fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.colorbar(label="Counts / 2.16e20 Muons")
plt.tight_layout()
plt.show()


plt.hist2d(trimmed_output_nue_array[:,0], trimmed_output_nue_array[:,1], weights=trimmed_output_nue_array[:,2],
                                                  bins=[enu_bins, theta_bins], cmap=my_cmap, norm=LogNorm())
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$E_{\nu_e}$ [MeV]", fontsize=16)
plt.ylabel(r"$\theta_{\nu_e}$ [rad]", fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.colorbar(label="Counts / 2.16e20 Muons")
plt.tight_layout()
plt.show()


np.savetxt("data/numu_flux_MeV_rad_SqrtS-10TeV_216e19Mu_25m2_10m_200m_v2.txt", trimmed_output_numu_array)
np.savetxt("data/nuebar_flux_MeV_rad_SqrtS-10TeV_216e19Mu_25m2_10m_200m_v2.txt", trimmed_output_nue_array)