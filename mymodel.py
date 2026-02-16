from extract_data import *
import celerite2
from celerite2 import terms
import numpy as np
from scipy.stats import norm, t


num_qsos = 190
num_bands = 3
num_hyperparameters = 8

data = []
for i in range(num_qsos):
    for band in ["g", "r", "i"]:
        data.append(get_data(i, band, sanitise=True, plot=False))

num_params = num_hyperparameters + 4*num_qsos*num_bands

def prior_transform(us):
    """
    Parameters are mu, sigma, tau, jitter for each light curve
    """
    hypers = us[0:num_hyperparameters]
    hypers[0] = 0.0 + 40.0*hypers[0]         # Mu for mean magnitude
    hypers[1] = 5.0*hypers[1]                # Sigma for mean magnitude
    hypers[2] = -5.0 + 10.0*hypers[2]        # Mu for log10_beta
    hypers[3] = 5.0*hypers[3]                # Sigma for log10_beta
    hypers[4] = 15.0*hypers[4]               # Mu for log10_tau
    hypers[5] = 5.0*hypers[5]                # Sigma for log10_tau
    hypers[6] = -5.0 + 10.0*hypers[6]        # Mu for log10_jitter
    hypers[7] = 5.0*hypers[7]                # Sigma for log10_jitter

    qso_params_3d = us[num_hyperparameters:].copy()
    qso_params_3d = qso_params_3d.reshape((num_qsos, num_bands, 4))

    qso_params_3d[:, :, 0] = norm.ppf(qso_params_3d[:, :, 0],
                                      loc=hypers[0], scale=hypers[1]) # Mean magnitude
    qso_params_3d[:, :, 1] = norm.ppf(qso_params_3d[:, :, 1],
                                      loc=hypers[2], scale=hypers[3]) # log10_beta in magnitudes
    qso_params_3d[:, :, 2] = norm.ppf(qso_params_3d[:, :, 2],
                                      loc=hypers[4], scale=hypers[5]) # log10_tau in days
    qso_params_3d[:, :, 3] = norm.ppf(qso_params_3d[:, :, 3],
                                      loc=hypers[6], scale=hypers[7]) # log10_jitter in magnitudes

    return np.hstack([hypers, qso_params_3d.flatten()])

def log_likelihood(params):

    logl = 0.0

    qso_params_3d = params[num_hyperparameters:].reshape((num_qsos, num_bands, 4))

    mu =    qso_params_3d[:, :, 0]   
    beta =  10.0**qso_params_3d[:, :, 1]
    tau   = 10.0**qso_params_3d[:, :, 2]
    jitter= 10.0**qso_params_3d[:, :, 3]
    sigma = beta*np.sqrt(0.5*tau)

    k = 0
    for i in range(num_qsos):
        for j in range(num_bands):

            try:
                term = terms.RealTerm(a=sigma[i, j]**2, c=1.0/tau[i, j])
                kernel = term
                gp = celerite2.GaussianProcess(kernel, mean=mu[i, j])
                this_data = data[k]["light_curve"]
                gp.compute(this_data[:,0], yerr=np.sqrt(this_data[:,2]**2 + jitter[i, j]**2))
                logl += gp.log_likelihood(this_data[:,1])
            except:
                logl += -1.0E300
            k += 1

    return logl


def both(us):
    return log_likelihood(prior_transform(us))

