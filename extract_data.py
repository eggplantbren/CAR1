import astropy
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

# Open the FITS file
fits_file = fits.open("TotalDat.fits")
totaldat = fits_file[1].data

def remove_outliers(data):
    median = np.median(data[:,1])
    q75, q25 = np.quantile(data[:,1], [0.75, 0.25])
    iqr = q75 - q25
    lower = median - 5*iqr
    upper = median + 5*iqr
    is_outlier = (data[:,1] + data[:,2] < lower) | (data[:,1] - data[:,2] > upper)
    print("{n} outliers removed.".format(n=np.sum(is_outlier)))
    return data[~is_outlier, :]

def get_data(qso_number, band, center_data=False, plot=True,
                deredshift=False, sanitise=False):
    """
    Band must be 'g', 'r' or 'i'.
    """

    t = totaldat[f"MJD_{band}"][qso_number, :]
    y = totaldat[f"MAG_{band}"][qso_number, :]
    err = totaldat[f"MAG_ERR_{band}"][qso_number, :]
    z = totaldat[f"Z"][qso_number]
    lbol = totaldat[f"log_LBOL"][qso_number]
    lamb = 1.0/(1.0 + z)
    if band == "g":
        lamb *= 4720.0
    elif band == "r":
        lamb *= 6415.0
    elif band == "i":
        lamb *= 7835.0

    keep = ~np.isnan(t)
    t, y, err = t[keep], y[keep], err[keep]

    ## Center the data
    if center_data:
        w = 1.0/err
        y -= np.sum(w*y)/np.sum(w)

    # De-redshift by scaling time axis
    if deredshift:
        t = t/(1.0 + z)

    data = np.empty((len(t), 3))
    data[:,0] = t
    data[:,1] = y
    data[:,2] = err
    if sanitise:
        data = remove_outliers(data)

    np.savetxt("data.txt", data)

    # Get reported tau values
#    log_tau = totaldat[f"log_TAU_OBS_{band}"][qso_number]
#    lower = totaldat[f"log_TAU_OBS_{band}_ERR_L"][qso_number]
#    upper = totaldat[f"log_TAU_OBS_{band}_ERR_U"][qso_number]

    if plot:
        plt.errorbar(data[:,0], data[:,1], yerr=data[:,2], fmt=".")
        plt.show()

    return dict(light_curve=data, redshift=z, lbol=lbol,
                log10_lambda=np.log10(lamb))

if __name__ == "__main__":

    f = open("qso_info.csv", "w")
    f.write("qso_number,redshift,log10_bol,log10_lambda\n")
    f.flush()
    for i in range(190):
        for color in ["g", "r", "i"]:
            data = get_data(i, color, sanitise=True, plot=False)

            f.write(str(i) + ",")
            f.write(str(data["redshift"]) + ",")
            f.write(str(data["lbol"]) + ",")
            f.write(str(data["log10_lambda"]))
            f.write("\n")
            f.flush()
    f.close()

fits_file.close()
