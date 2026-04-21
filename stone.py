from pylab import *
from mymodel import num_qsos, num_bands, num_hyperparameters

sample = loadtxt("sample.txt")

log10_tau = []
for i in range(int(0.3*sample.shape[0]), sample.shape[0]):
    params = sample[i, :]
    qso_params_3d = params[num_hyperparameters:].reshape((num_qsos, num_bands, 4))
    log10_tau_row = qso_params_3d[:, :, 2].flatten()
    log10_tau.append(log10_tau_row)
    print(i, flush=True)

log10_tau = array(log10_tau)
plt.plot(log10_tau.mean(axis=0))
plt.show()


import astropy
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

# Open the FITS file
fits_file = fits.open("TotalDat.fits")
totaldat = fits_file[1].data
stone_g = totaldat["log_TAU_OBS_g"]
stone_r = totaldat["log_TAU_OBS_r"]
stone_i = totaldat["log_TAU_OBS_i"]
stone_all = []
for i in range(num_qsos):
    stone_all.append(stone_g[i])
    stone_all.append(stone_r[i])
    stone_all.append(stone_i[i])

plt.plot(stone_all, log10_tau.mean(axis=0), ".")
x = linspace(2, 6, 1001)
plt.plot(x, x)
plt.xlabel("Stone")
plt.ylabel("Me")
plt.axis("equal")
plt.show()

