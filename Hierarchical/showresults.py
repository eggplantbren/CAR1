import dnest4.classic as dn4
import numpy as np
import matplotlib.pyplot as plt
import os


dn4.postprocess()

# Make posterior histogram of magnitudes
path = "../output/"
filenames = os.listdir(path)

my_bins = np.linspace(15.0, 25.0, 101)

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 14,
})


for f in filenames:
    posterior_samples = np.loadtxt(path + f)
    plt.hist(posterior_samples[:,0], bins=my_bins, color="k", alpha=0.3,
             histtype="step")

plt.xlabel("Magnitude")
plt.ylabel("Probability Density")
plt.show()
