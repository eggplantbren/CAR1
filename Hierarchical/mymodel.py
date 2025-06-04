import numpy as np
import os

num_params = 8

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

    # mu and sigma for magnitudes
    params[0] = 40.0*us[0]
    params[1] = 10.0**(-2.0 + 4.0*us[1])

    # mu and sigma for log10(sigma)
    params[2] = np.exp(np.log(1E-3) + np.log(1E4)*us[2])
    params[3] = 3.0*us[3]

    # mu and sigma for log10(tau)
    params[4] = np.exp(np.log(1.0)  + np.log(1E6)*us[4])
    params[5] = 3.0*us[5]

    # mu and sigma for log10(jitter)
    params[6] = np.exp(np.log(1E-3) + np.log(1E3)*us[6])
    params[7] = 3.0*us[7]

    return params

def log_likelihood(params):
    return -num_params*0.5*np.log(2*np.pi*0.01**2)-0.5*np.sum((params/0.01)**2)


def both(us):
    return log_likelihood(prior_transform(us))

