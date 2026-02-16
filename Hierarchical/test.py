import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt
from scipy.stats import norm

# True p(theta | alpha)
mu = 5.0
sigma = 2.0

# Parameters of individual-object prior
mu_pi = 3.0
sigma_pi = 5.0

# How far off is the likelihood function from the true parameter
sigma_lik = 1.0

# Number of objects and posterior samples per object
num_objects = 10
num_posterior_samples = 10000

# True parameters of objects
thetas = mu + sigma*rng.randn(num_objects)

# Generate posterior samples
posterior_samples = np.empty((num_objects, num_posterior_samples))

for i in range(num_objects):

    # Hypothetical gaussian likelihood function
    mu_lik = thetas[i] + sigma_lik*rng.randn(num_posterior_samples)

    # It'll be a gaussian posterior
    tau_post = 1.0/sigma_pi**2 + 1.0/sigma_lik**2
    var_post = 1.0/tau_post
    sigma_post = np.sqrt(var_post)
    mu_post = var_post*(mu_pi/sigma_pi**2 + mu_lik/sigma_lik**2)

    posterior_samples[i, :] = mu_post + sigma_post*rng.randn()



def logsumexp(xs):
    top = np.max(xs)
    return (top + np.log(np.sum(np.exp(xs - top))))

def logmeanexp(xs):
    return(logsumexp(xs) - np.log(len(xs)))


def log_conditional_prior(params, samples):

    # Just normal distributions
    return norm.logpdf(samples, loc=params[0], scale=params[1])



def log_interim_prior(samples):
    """
    Evaluate the interim prior at one set of samples.
    """
    return norm.logpdf(samples, loc=mu_pi, scale=sigma_pi)


def log_likelihood(params):
    logl = 0.0

    for i in range(posterior_samples.shape[0]):

        # The expectation inside the product sign
        loge = logmeanexp(log_conditional_prior(params, posterior_samples[i, :]) -
                          log_interim_prior(posterior_samples[i, :]))

        logl = logl + loge

    return logl

