from extract_data import *
import celerite2
from celerite2 import terms
import numpy as np
from scipy.stats import t


num_qsos = 190
num_bands = 3
num_hyperparameters = 0

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
    params = us[num_hyperparameters:]
    params = params.reshape((num_qsos, num_bands, 4))

    for i in range(num_qsos):
        for j in range(num_bands):
            params[i, j, 0] = 20.0 + 5.0*t.ppf(params[i, j, 0], df=4) # Mean magnitude
            params[i, j, 1] = 0.0  + 5.0*t.ppf(params[i, j, 1], df=4) # log10_beta in magnitudes
            params[i, j, 2] = 0.0  + 5.0*t.ppf(params[i, j, 2], df=4) # log10_tau in days
            params[i, j, 3] = 0.0  + 5.0*t.ppf(params[i, j, 3], df=4) # log10_jitter in magnitudes

    return params

def log_likelihood(params):

    logl = 0.0
    k = 0
    for i in range(num_qsos):
        for j in range(num_bands):
            mu =    params[i, j, 0]   
            beta =  10.0**params[i, j, 1]
            tau   = 10.0**params[i, j, 2]
            jitter= 10.0**params[i, j, 3]
            sigma = beta*np.sqrt(0.5*tau)

            try:
                term = terms.RealTerm(a=sigma**2, c=1.0/tau)
                kernel = term
                gp = celerite2.GaussianProcess(kernel, mean=mu)
                gp.compute(data[k][:,0], yerr=np.sqrt(data[k][:,2]**2 + jitter**2))
                logl += gp.log_likelihood(data[k][:,1])
            except:
                logl += -1.0E300
            k += 1

    return logl


def both(us):
    return log_likelihood(prior_transform(us))

