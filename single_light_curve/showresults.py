import dnest4.classic as dn4
dn4.postprocess()

import corner
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 14,
})

ndim = 5

posterior_sample = np.loadtxt("posterior_sample.txt")

# Compute log10_sigma from log10_beta
log10_sigma = posterior_sample[:,1] + 0.5*(posterior_sample[:,2] - np.log10(2.0))
posterior_sample = np.column_stack([posterior_sample, log10_sigma])
posterior_sample = posterior_sample[:, [0, 1, 4, 2, 3]]
print(posterior_sample.shape)

figure = corner.corner(posterior_sample,
    labels=["$\\mu$", "$\\log_{10}(\\beta)$", "$\\log_{10}(\\sigma)$",
            "$\\log_{10}(\\tau)$",
            "$\\log_{10}$(jitter)"],
            plot_contours=False,
            plot_density=False, fontsize=14,
            hist_kwargs={"color":"blue", "alpha":0.2, "histtype":"stepfilled",
                         "edgecolor":"black","lw":"3"} )

axes = np.array(figure.axes).reshape((ndim, ndim))

for i in range(ndim):
	ax = axes[i,i]

plt.show()
