import celerite2
from celerite2 import terms
import numpy as np
from extract_data import *
from scipy.stats import norm

num_params = 6 + 190*3*3
all_data = get_all_data()

def prior_transform(us):
    """
    Parameters are mu, sigma, tau, jitter
    """
    params = us.copy()

    # The first three are hyperparameters now
    params[0] = 0.0 + 40.0*us[0] # Mean magnitude
    params[1] = np.exp(np.log(1E-3) + np.log(1E4)*us[1]) # Sigma in magnitudes
    params[2] = np.exp(np.log(1.0)  + np.log(1E6)*us[2]) # Tau in days

    params[3] = 10.0*us[3] # Diversity of magnitudes
    params[4] = 2.0*us[4] # Diversity of log10-sigmas
    params[5] = 2.0*us[5] # Diversity of log10-taus

    k = 6
    for i in range(190):
        for j in range(3):
            # Magnitude
            params[k] = params[0] + params[3]*norm.ppf(us[k])
            k += 1

            # Sigma
            params[k] = params[1]*10.0**(params[4]*norm.ppf(us[k]))
            k += 1

            # Tau
            params[k] = params[2]*10.0**(params[5]*norm.ppf(us[k]))

    return params

def log_likelihood(params):

    k = 6
    logl = 0.0
    m = 0
    for i in range(190):
        for j in range(3):
            mu, sigma, tau = params[k:k+3]
            k += 3

            term = terms.RealTerm(a=sigma**2, c=1.0/tau)
            kernel = term

            try:
                gp = celerite2.GaussianProcess(kernel, mean=mu)
                gp.compute(all_data[m][:,0], yerr=all_data[m][:,2])
            except:
                return -1E300

            logl += gp.log_likelihood(all_data[m][:,1])
            m += 1
            
    return logl


def both(us):
    return log_likelihood(prior_transform(us))

