import numpy as np
import os
import pandas as pd

num_params = 11

# Load quasar metadata
qso_info = pd.read_csv("../qso_info.csv")
all_samples = []

# Construct the posterior sample filenames and attempt to load them all,
# in the same order as they appear in qso_info.csv.
success = []
for i in range(qso_info.shape[0]):
    try:
        qso_number, band = qso_info.iloc[i, 0:2]
        filename = "../output/posterior_sample_" + str(qso_number) + "_" + str(band) + ".txt"
        samples = np.loadtxt(filename)
        all_samples.append(samples)
        success.append(i)
        print("Loaded samples from file " + filename, flush=True)
    except:
        pass

qso_info = qso_info.iloc[success, :]
assert(qso_info.shape[0] == len(all_samples))

mean_log10_lbol = np.mean(qso_info["log10_lbol"])
mean_log10_lambda = np.mean(qso_info["log10_lambda"])

def logsumexp(xs):
    top = np.max(xs)
    return (top + np.log(np.sum(np.exp(xs - top))))

def logmeanexp(xs):
    return(logsumexp(xs) - np.log(len(xs)))




def prior_transform(us):
    
    params = np.empty(us.size)

    # mu and sigma for magnitudes
    params[0] = 40.0*us[0]
    params[1] = 3.0*us[1]

    # mu and sigma for log10(sigma)
    params[2] = -3.0 + 4.0*us[2]
    params[3] = 3.0*us[3]

    # mu and sigma for log10(tau)
    params[4] = 0.0 + 6.0*us[4]
    params[5] = 3.0*us[5]

    # mu and sigma for log10(jitter)
    params[6] = -3.0 + 3.0*us[6]
    params[7] = 3.0*us[7]

    # Regression coefficients
    params[8] = -10.0 + 20.0*us[8] # Coefficient for log10_lbol
    params[9] = -10.0 + 20.0*us[9] # Coefficient for log10_lambda
    params[10] = -1.0 + 2.0*us[10] # Coefficient for interaction term

    return params


def log_conditional_prior(params, samples, i):

    # Just normal distributions
    logp = 0.0

    logp += -0.5*np.log(2.0*np.pi*params[1]**2) \
                    - 0.5*(samples[:,0] - params[0])**2/params[1]**2

    logp += -0.5*np.log(2.0*np.pi*params[3]**2) \
                    - 0.5*(samples[:,1] - params[2])**2/params[3]**2

    # Compute expected value of log10_tau according to regression model
    mu = params[4] + 1.0*np.log10(1.0 + qso_info["redshift"][i]) \
                   + params[8]*(qso_info["log10_lbol"][i] - mean_log10_lbol) \
                   + params[9]*(qso_info["log10_lambda"][i] - mean_log10_lambda) \
                   + params[10]*(qso_info["log10_lbol"][i] - mean_log10_lbol) \
                               *(qso_info["log10_lambda"][i] - mean_log10_lambda)

    logp += -0.5*np.log(2.0*np.pi*params[5]**2) \
                    - 0.5*(samples[:,2] - mu)**2/params[5]**2

    logp += -0.5*np.log(2.0*np.pi*params[7]**2) \
                    - 0.5*(samples[:,3] - params[6])**2/params[7]**2

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

    # mu
    logp += -0.5*np.log(2.0*np.pi*10.0**2) \
                    - 0.5*(samples[:,0] - 20.0)**2/10.0**2

    # log10_sigma
    logp += -0.5*np.log(2.0*np.pi*5.0**2) \
                    - 0.5*(samples[:,1] - 0.0)**2/5.0**2

    # log10_tau
    logp += -0.5*np.log(2.0*np.pi*5.0**2) \
                    - 0.5*(samples[:,2] - 0.0)**2/5.0**2

    # log10_jitter
    logp += -0.5*np.log(2.0*np.pi*5.0**2) \
                    - 0.5*(samples[:,3] - 0.0)**2/5.0**2
 
    return logp


def log_likelihood(params):
    logl = 0.0

    i = 0
    for samples in all_samples:

        # The expectation inside the product sign
        loge = logmeanexp(log_conditional_prior(params, samples, i) -
                          log_interim_prior(params, samples))

        logl = logl + loge
        i += 1

    return logl

def both(us):
    return log_likelihood(prior_transform(us))

