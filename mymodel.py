import celerite2
from celerite2 import terms
import numpy as np

num_params = 4
data = np.loadtxt("data.txt")

def prior_transform(us):
    """
    Parameters are mu, sigma, tau, jitter
    """
    params = us.copy()
    params[0] = 0.0 + 40.0*us[0] # Mean magnitude
    params[1] = -5.0 + 15.0*us[1] # log10_sigma in magnitudes
    params[2] = -5.0 + 15.0*us[2] # log10_tau in days
    params[3] = -5.0 + 15.0*us[3] # log10_jitter in magnitudes
    return params

def log_likelihood(params):
    mu, log10_sigma, log10_tau, log10_jitter = params
    sigma = 10.0**log10_sigma
    tau   = 10.0**log10_tau
    jitter= 10.0**log10_jitter

    try:
        term = terms.RealTerm(a=sigma**2, c=1.0/tau)
        kernel = term
        gp = celerite2.GaussianProcess(kernel, mean=mu)
        gp.compute(data[:,0], yerr=np.sqrt(data[:,2]**2 + jitter**2))
        logl = gp.log_likelihood(data[:,1])
    except:
        logl = -1.0E300

    return logl


def both(us):
    return log_likelihood(prior_transform(us))

