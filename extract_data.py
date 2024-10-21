import astropy
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

# Open the FITS file
fits_file = fits.open("TotalDat.fits")
totaldat = fits_file[1].data

qso_number = 0
band = "g"

t = totaldat[f"MJD_{band}"][qso_number, :]
y = totaldat[f"MAG_{band}"][qso_number, :]
err = totaldat[f"MAG_ERR_{band}"][qso_number, :]

keep = ~np.isnan(t)
t, y, err = t[keep], y[keep], err[keep]

## Center the data
#w = 1.0/err
#y -= np.sum(w*y)/np.sum(w)

data = np.empty((len(t), 3))
data[:,0] = t
data[:,1] = y
data[:,2] = err
np.savetxt("data.txt", data)

# Get reported tau values
log_tau = totaldat[f"log_TAU_OBS_{band}"][qso_number]
lower = totaldat[f"log_TAU_OBS_{band}_ERR_L"][qso_number]
upper = totaldat[f"log_TAU_OBS_{band}_ERR_U"][qso_number]

print(log_tau, lower, upper)

plt.errorbar(t, y, yerr=err, fmt=".")
plt.show()

fits_file.close()

