import dnest4.classic as dn4
dn4.postprocess()

import corner
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 14,
})

ndim = 4

posterior_sample = np.loadtxt("posterior_sample.txt")

figure = corner.corner(posterior_sample,
    labels=["$\\mu$", "$\\log_{10}(\\sigma)$", "$\\log_{10}(\\tau)$",
            "$\\log_{10}$(jitter)"],
            plot_contours=False,
            plot_density=False, fontsize=14,
            hist_kwargs={"color":"blue", "alpha":0.3, "histtype":"stepfilled",
                         "edgecolor":"black","lw":"3"} )

axes = np.array(figure.axes).reshape((ndim, ndim))

for i in range(ndim):
	ax = axes[i,i]

plt.show()
