import celerite2
from celerite2 import terms
import numpy as np

num_params = 3
data = np.loadtxt("data.txt")

def prior_transform(us):
    """
    Parameters are mu, sigma, tau
    """
    params = us.copy()
    params[0] = 0.0 + 40.0*us[0]
    params[1] = np.exp(np.log(1E-3) + np.log(1E4)*us[1])
    params[2] = np.exp(np.log(1.0)  + np.log(1E6)*us[2])
    return params

def log_likelihood(params):
    mu, sigma, tau = params
    term = terms.RealTerm(a=sigma**2, c=1.0/tau)
    kernel = term
    gp = celerite2.GaussianProcess(kernel, mean=mu)
    gp.compute(data[:,0], yerr=data[:,2])
    return gp.log_likelihood(data[:,1])


def both(us):
    return log_likelihood(prior_transform(us))

