import numpy as np
import os


num_params = 10

# Load posterior samples
files = os.listdir("../output/")
all_samples = []
for f in files:
    all_samples.append(np.loadtxt("../output/" + f))


def logsumexp(xs):
    top = np.max(xs)
    return (top + np.log(np.sum(np.exp(xs - top))))

def logmeanexp(xs):
    return(logsumexp(xs) - np.log(len(xs)))



def prior_transform(us):
    return us - 0.5

def log_likelihood(params):
    return -num_params*0.5*np.log(2*np.pi*0.01**2)-0.5*np.sum((params/0.01)**2)


def both(us):
    return log_likelihood(prior_transform(us))

