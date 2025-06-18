import dnest4.classic as dn4
import numpy as np
import matplotlib.pyplot as plt
import os


dn4.postprocess()

# Make posterior histogram of magnitudes
path = "../output/"
filenames = os.listdir(path)

my_bins = np.linspace(15.0, 25.0, 51)
x = np.linspace(15.0, 25.0, 5001)

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 14,
})

plt.figure(figsize=(11, 7))

means = []

for f in filenames:
    posterior_samples = np.loadtxt(path + f)
    means.append(np.mean(posterior_samples[:,0]))

plt.hist(means, bins=my_bins, color="k", alpha=0.7, linewidth=2, density=True,
         histtype="step", label="Individual Object Posterior Means")


posterior_sample = np.loadtxt("posterior_sample.txt")
for i in range(min(100, posterior_sample.shape[0])):
    mu, sigma = posterior_sample[i, 0:2]
    y = np.exp(-0.5*(x - mu)**2/sigma**2)/np.sqrt(2.0*np.pi*sigma**2)
    if i==0:
        label = "Possible $f(\\mu | \\alpha)$"
    else:
        label = None
    plt.plot(x, y, "b", alpha=0.1, label=label)

plt.xlabel("Magnitude $\\mu$")
plt.ylabel("Probability Density")
plt.legend()
plt.show()
