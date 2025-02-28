import numpy as np
import matplotlib.pyplot as plt


from matplotlib.pylab import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


from scipy.signal import savgol_filter


# Load the MultiNest output
# Assuming the output file has 4 columns corresponding to the parameters

#multinest_output = np.loadtxt("multinest/SR/sqrts-500GeV_SR_.txt")
multinest_output = np.loadtxt("multinest/MC/sqrts-3TeV_MC_.txt")

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



def cornerplot(samples, deltaChi2, prob, sm_values, labels,
               axes_ranges=[(0.0, 1.0), (0.0, 1.0), (-0.3, 0.3), (0.0, 0.3)]):

    p_sorted_idx = np.argsort(prob)


    # color code
    colors = np.zeros_like(deltaChi2, dtype=object)

    # delta Chi^2 values are based on 4 DOF
    colors[deltaChi2 < 4.71] = 'navy'
    colors[(deltaChi2 >= 4.71) & (deltaChi2 < 9.715)] = 'mediumseagreen'
    colors[deltaChi2 >= 9.715] = 'silver'  # Optional: color values greater than 9.715 red, or keep this line out

    colors2 = np.zeros_like(deltaChi2, dtype=object)
    colors2[deltaChi2 < 4.71] = 'b'
    colors2[(deltaChi2 >= 4.71) & (deltaChi2 < 9.715)] = 'mediumseagreen'
    colors2[deltaChi2 >= 9.715] = 'silver'  # Optional: color values greater than 9.715 red, or keep this line out

    alphas = np.zeros_like(deltaChi2, dtype=object)
    alphas[deltaChi2 < 4.71] = 0.5
    alphas[(deltaChi2 >= 4.71) & (deltaChi2 < 9.715)]  = 0.9
    alphas[deltaChi2 >= 9.715] = 0.5

    # ranges and SM values
    bins = [np.linspace(r[0], r[1], 50) for r in axes_ranges]


    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    # Loop over pairs of parameters to create the contour plots
    for i in range(4):
        for j in range(4):
            axes[i, j].grid(True, linestyle='-', linewidth=0.5, alpha=0.3, color='gray', dashes=(5,10))
            axes[i, j].tick_params(axis="x", labelsize=16)
            axes[i, j].tick_params(axis="y", labelsize=16)
            for line in plt.gca().xaxis.get_gridlines():
                line.set_dash_capstyle('round')
            
            if i == j:
                # Diagonal: plot the histograms of the individual parameters
                bin_centers = (bins[i][1:] + bins[i][:-1])/2
                h = np.histogram(samples[:,i], weights=prob, bins=bins[i])[0]
                smoothed_h = savgol_filter(h, 10, 2)

                axes[i, j].hist(bin_centers, bins=bins[i], weights=smoothed_h, density=True,
                                color='blue', alpha=0.7, histtype='step')
                
                # Plotting direct prob curve
                # sorted_indices = np.argsort(samples[:, i])
                # axes[i,j].plot(samples[:, i][sorted_indices], savgol_filter(prob[sorted_indices], 500, 1))
                axes[i,j].set_xlim(axes_ranges[j])
                axes[i,j].set_ylim(bottom=0.0)
                axes[i,j].set_ylabel(r"Probability Density")

            elif i > j:
                # Lower triangle: plot contours of constant log-likelihood
                axes[i, j].scatter(samples[:, j][p_sorted_idx], samples[:, i][p_sorted_idx],
                                c=colors[p_sorted_idx], cmap='viridis', s=1, alpha=alphas, marker='.')
                #axes[i, j].contour(np.histogram2d(samples[:, j], samples[:, i], bins=gLR_bins, weights=deltaChi2)[0], levels=[2.30, 6.18, 11.83, 200.0, 1000.0], colors='white')
                axes[i,j].set_ylim(axes_ranges[i])
                axes[i,j].set_xlim(axes_ranges[j])

                axes[i,j].plot(sm_values[j], sm_values[i], marker='*', color='r')
            else:
                #axes[i,j].plot(sm_values[j], sm_values[i], marker='*', color='r')
                axes[i, j].axis('off')

            # Set parameter labels
            if i == 3:
                axes[i, j].set_xlabel(labels[j], fontsize=16)
            #else:
            #    axes[i, j].set_xticks([])
            if j == 0 and i != j:
                axes[i, j].set_ylabel(labels[i], fontsize=16)

    plt.tight_layout()

    plt.show()
    plt.close()


def plot_gv_gA():
    # plot the gV - gA plane

    g_nue = multinest_output[:, 2]
    g_numu = multinest_output[:, 3]
    g_L = multinest_output[:, 4]
    g_R = multinest_output[:, 5]

    gVe = 2*g_nue*(g_L + g_R)
    gAe = 2*g_nue*(g_L - g_R)
    gVmu = 2*g_nue*(g_L + g_R)
    gAmu = 2*g_nue*(g_L - g_R)
    plt.scatter(gVe, gAe, c=colors, alpha=alphas, cmap='viridis', s=1)
    plt.scatter(gVmu, gAmu, c=colors2, alpha=alphas, cmap='viridis', s=1)

    # SM point
    plt.plot(-0.5 + 2*fluxes_dict["500GeV"]["SSW"], -0.5, color='r', marker="*")

    plt.xlabel(r"$g_V$", fontsize=16)
    plt.ylabel(r"$g_A$", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim((-0.6, 0.6))
    plt.ylim((-0.6, 0.6))
    plt.show()
    plt.close()


def main():
    # Extract the parameter columns
    g_nue = multinest_output[:, 2]
    g_numu = multinest_output[:, 3]
    g_L = multinest_output[:, 4]
    g_R = multinest_output[:, 5]


    deltaChi2 = multinest_output[:,1]
    prob = multinest_output[:,0]
    sm_values = [0.5, 0.5, -0.5 + fluxes_dict["500GeV"]["SSW"], fluxes_dict["500GeV"]["SSW"]]

    # Combine the parameters into a single array for corner plot
    samples = np.vstack([g_nue, g_numu, g_L, g_R]).T

    # Define the labels for the parameters
    labels = [r"$g_{\nu_e}$", r"$g_{\nu_\mu}$", r"$g_L^e$", r"$g_R^e$"]

    axes_ranges = [(0.0, 1.0), (0.0, 1.0), (-0.3, 0.3), (0.0, 0.3)]

    cornerplot(samples, deltaChi2, prob, sm_values, labels, axes_ranges)


    # plot the products: gL*gnue, gR*gnue, gL*gnumu, gR*gnumu
    gLe = 2*g_nue*g_L
    gRe = 2*g_nue*g_R
    gLmu = 2*g_numu*g_L
    gRmu = 2*g_numu*g_R
    samples = np.vstack([gLe, gRe, gLmu, gRmu]).T
    sm_values = [-0.5 + fluxes_dict["500GeV"]["SSW"], fluxes_dict["500GeV"]["SSW"],
                -0.5 + fluxes_dict["500GeV"]["SSW"], fluxes_dict["500GeV"]["SSW"]]
    labels = [r"$2 g_{\nu_e} g_L^e$", r"$2 g_{\nu_e} g_R^e$", r"$2 g_{\nu_\mu} g_L^e$", r"$2 g_{\nu_\mu} g_R^e$"]
    axes_ranges = [(-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3)]
    cornerplot(samples, deltaChi2, prob, sm_values, labels, axes_ranges)



if __name__ == "__main__":
    main()

