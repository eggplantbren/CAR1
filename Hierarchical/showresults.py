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

    assert(mode in ["mu", "sigma", "tau"])

    means = []

    # These are columns in the individual object files
    if mode == "mu":
        column = 0
    elif mode == "sigma":
        column = 1
    elif mode == "tau":
        column = 2

    for f in filenames:
        posterior_samples = np.loadtxt(path + f)
        means.append(np.median(posterior_samples[:,column]))

    if mode in ["sigma", "tau"]:
        means = np.log10(means)

    plt.hist(means, bins=50, color="k", alpha=0.7, linewidth=2, density=True,
             histtype="step", label="Individual Object Posterior Medians")

    # Convert to column number for the hyperparameter posterior sample file
    column *= 2

    
    xmin = np.min(means) - 0.1*np.ptp(means)
    xmax = np.max(means) + 0.1*np.ptp(means)
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

plt.figure(figsize=(8, 12))
plt.subplot(3, 1, 1)
make_plot("mu")

plt.subplot(3, 1, 2)
make_plot("sigma")

plt.subplot(3, 1, 3)
make_plot("tau")
plt.savefig("results.pdf", bbox_inches="tight")
plt.show()


