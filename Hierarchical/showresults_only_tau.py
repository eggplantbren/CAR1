import dnest4.classic as dn4
import numpy as np
import matplotlib.pyplot as plt
import os


dn4.postprocess()

# Make posterior histogram of magnitudes
path = "../output/"
filenames = os.listdir(path)

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 11,
})


def make_plot(mode):

    medians = []

    # These are columns in the individual object files
    column = 2

    for f in filenames:
        posterior_samples = np.loadtxt(path + f)
        medians.append(np.median(posterior_samples[:,column]))

    if mode in ["sigma", "tau"]:
        medians = np.log10(medians)

    plt.hist(medians, bins=50, color="k", alpha=0.7, linewidth=2, density=True,
             histtype="step", label="Individual Object Posterior Medians")

    # Column number for the hyperparameter posterior sample file
    column = 0

    
    xmin = np.min(medians) - 0.1*np.ptp(medians)
    xmax = np.max(medians) + 0.1*np.ptp(medians)
    x = np.linspace(xmin, xmax, 5001)



    if mode == "mu":
        symbol = "\\mu"
        xlabel = "Magnitude $\\mu$"
    elif mode == "sigma":
        symbol = "\\sigma"
        xlabel = "$\\log_{10}(\\sigma)$"
    elif mode == "tau":
        symbol = "\\tau"
        xlabel = "$\\log_{10}(\\tau)$"

    posterior_sample = np.loadtxt("posterior_sample.txt")
    for i in range(min(100, posterior_sample.shape[0])):
        mu, sigma = posterior_sample[i, column:column+2]
        y = np.exp(-0.5*(x - mu)**2/sigma**2)/np.sqrt(2.0*np.pi*sigma**2)
        if i==0:
            label = f"Possible $f({symbol} | \\alpha)$"
        else:
            label = None
        plt.plot(x, y, "b", alpha=0.1, label=label)



    plt.xlabel(xlabel)
    plt.ylabel("Probability Density")
    plt.legend()

make_plot("tau")
plt.savefig("results_only_tau.pdf", bbox_inches="tight")
plt.show()

