import celerite2
from celerite2 import terms
import numpy as np
from scipy.stats import t

num_params = 4
data = np.loadtxt("data.txt")

def prior_transform(us):
    """
    Parameters are mu, sigma, tau, jitter
    """
    params = us.copy()
    params[0] = 20.0 + 10.0*t.ppf(us[0], df=4) # Mean magnitude
    params[1] = 0.0  + 5.0*t.ppf(us[1], df=4)  # log10_beta in magnitudes
    params[2] = 0.0  + 5.0*t.ppf(us[2], df=4)  # log10_tau in days
    params[3] = 0.0  + 5.0*t.ppf(us[3], df=4)  # log10_jitter in magnitudes
    return params

def log_likelihood(params):
    mu, log10_beta, log10_tau, log10_jitter = params
    beta = 10.0**log10_beta
    tau   = 10.0**log10_tau
    jitter= 10.0**log10_jitter
    sigma = beta*np.sqrt(0.5*tau)

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

