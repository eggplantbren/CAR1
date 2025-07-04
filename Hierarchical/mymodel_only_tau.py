import numpy as np
import os

num_params = 2

# Load posterior samples
files = os.listdir("../output/")
all_samples = []
for f in files:
    samples = np.loadtxt("../output/" + f)

    # Convert all but the first column to log10
    samples[:, 1:] = np.log10(samples[:, 1:])

    all_samples.append(samples)


def logsumexp(xs):
    top = np.max(xs)
    return (top + np.log(np.sum(np.exp(xs - top))))

def logmeanexp(xs):
    return(logsumexp(xs) - np.log(len(xs)))



def prior_transform(us):
    
    params = np.empty(us.size)

    # mu and sigma for log10(tau)
    params[0] = 0.0 + 6.0*us[0]
    params[1] = 3.0*us[1]

    return params


def log_conditional_prior(params, samples):

    # Just normal distributions
    logp = 0.0

    logp += -0.5*np.log(2.0*np.pi*params[1]**2) \
                    - 0.5*(samples[:,2] - params[0])**2/params[1]**2

    return logp

def log_interim_prior(params, samples):
    """
    Evaluate the interim prior at one set of samples.
    This must match the prior in
    ../mymodel.py. REMEMBERING that we are using log10 here for some
    columns. NOTE: currently un-normalised, so marginal likelihood
    estimate not right.
    """
    logp = 0.0

    logp += -0.5*np.log(2.0*np.pi*5.0**2) \
                    - 0.5*(samples[:,2] - 0.0)**2/5.0**2
 
    return logp


def log_likelihood(params):
    logl = 0.0

    for samples in all_samples:

        # The expectation inside the product sign
        loge = logmeanexp(log_conditional_prior(params, samples) -
                          log_interim_prior(params, samples))

        logl = logl + loge

    return logl

def both(us):
    return log_likelihood(prior_transform(us))

