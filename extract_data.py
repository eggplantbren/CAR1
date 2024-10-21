import astropy
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

# Open the FITS file
fits_file = fits.open("TotalDat.fits")
totaldat = fits_file[1].data

t = totaldat["MJD_g"][0, :]
y = totaldat["MAG_g"][0, :]
err = totaldat["MAG_ERR_g"][0, :]

keep = ~np.isnan(t)
t, y, err = t[keep], y[keep], err[keep]

data = np.empty((len(t), 3))
data[:,0] = t
data[:,1] = y
data[:,2] = err
np.savetxt("data.txt", data)

# Get reported tau values
log_tau = totaldat["log_TAU_OBS_g"][0]
lower = totaldat["log_TAU_OBS_g_ERR_L"][0]
upper = totaldat["log_TAU_OBS_g_ERR_U"][0]

print(log_tau, lower, upper)

plt.errorbar(t, y, yerr=err, fmt=".")
plt.show()

fits_file.close()

