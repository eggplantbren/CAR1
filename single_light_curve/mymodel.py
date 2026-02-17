import celerite2
from celerite2 import terms
import numpy as np

data = np.loadtxt("../data.txt")
num_params = 4

def prior_transform(us):
    """
    Parameters are mu, sigma, tau, jitter for each light curve
    """

    params = us.copy()
    params[0] = 40.0*us[0]        # mu
    params[1] = -5.0 + 10.0*us[1] # log10_beta
    params[2] = 15.0*us[2]        # log10_tau
    params[3] = -5.0 + 10.0*us[3] # log10_jitter

    return params

def log_likelihood(params):

    logl = 0.0

    mu = params[0]
    beta, tau, jitter = 10.0**params[1:4]
    sigma = beta*np.sqrt(0.5*tau)

    try:
        term = terms.RealTerm(a=sigma**2, c=1.0/tau)
        kernel = term
        gp = celerite2.GaussianProcess(kernel, mean=mu)
        gp.compute(data[:,0], yerr=np.sqrt(data[:,2]**2 + jitter**2))
        logl = gp.log_likelihood(data[:,1])
    except Exception:
        logl = -1.0E300

    return logl


def both(us):
    return log_likelihood(prior_transform(us))

