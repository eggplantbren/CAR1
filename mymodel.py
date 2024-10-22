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
    params[1] = np.exp(np.log(1E-3) + np.log(1E4)*us[1]) # Sigma in magnitudes
    params[2] = np.exp(np.log(1.0)  + np.log(1E6)*us[2]) # Tau in days
    params[3] = np.exp(np.log(1E-3) + np.log(1E3)*us[3]) # Jitter in magnitudes
    return params

def log_likelihood(params):
    mu, sigma, tau, jitter = params
    term = terms.RealTerm(a=sigma**2, c=1.0/tau)
    kernel = term
    gp = celerite2.GaussianProcess(kernel, mean=mu)
    gp.compute(data[:,0], yerr=np.sqrt(data[:,2]**2 + jitter**2))
    return gp.log_likelihood(data[:,1])


def both(us):
    return log_likelihood(prior_transform(us))

