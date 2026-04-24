import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 14,
})

posterior_sample = np.loadtxt("posterior_sample.txt")
plt.hist(posterior_sample[:,2], 50, density=True, alpha=0.3,
         label="Fixed $\\mu$")

posterior_sample = np.loadtxt("posterior_sample_orig.txt")
plt.hist(posterior_sample[:,2], 50, density=True, alpha=0.3,
         label="Free $\\mu$")

plt.xlabel("$\\log_{10}(\\tau/{\\rm days})$")
plt.ylabel("Probability Density")
plt.xlim([2.0, 10.0])
plt.axvline(4.565956996103984, color="k", alpha=0.6, label="True value")
plt.legend()

plt.savefig("two_posteriors.pdf")
plt.show()

