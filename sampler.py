# A custom metropolis sampler for the hierarchical model.
import numpy as np
import numpy.random as rng

def wrap(x, a, b):
    L = b - a
    return a + (x - a) - L * np.floor((x - a) / L)

num_hyperparameters = 8
num_qsos = 190
num_bands = 3

# Hyperparameter prior boundaries
bounds = np.array([[0.0, 40.0],
                   [0.0, 5.0],
                   [-5.0, 5.0],
                   [0.0, 5.0],
                   [0.0, 15.0],
                   [0.0, 5.0],
                   [-5.0, 5.0],
                   [0.0, 5.0]])
h_idx = dict(mu_mu=0, sig_mu=1,
              mu_log10_beta=2, sig_log10_beta=3,
              mu_log10_tau=4, sig_log10_tau=5,
              mu_log10_jitter=6, sig_log10_jitter=7)

ranges = bounds[:,1] - bounds[:,0]

assert bounds.shape[0] == num_hyperparameters

class Particle:
    def __init__(self):
        self.hypers = bounds[:,0] + ranges*rng.rand(num_hyperparameters)

        # QSO parameters
        self.mu = self.hypers[h_idx["mu_mu"]] + self.hypers[h_idx["sig_mu"]]*rng.randn(num_qsos, num_bands)
        self.log10_beta = self.hypers[h_idx["mu_log10_beta"]] + self.hypers[h_idx["sig_log10_beta"]]*rng.randn(num_qsos, num_bands)
        self.log10_tau = self.hypers[h_idx["mu_log10_tau"]] + self.hypers[h_idx["sig_log10_tau"]]*rng.randn(num_qsos, num_bands)
        self.log10_jitter = self.hypers[h_idx["mu_log10_jitter"]] + self.hypers[h_idx["sig_log10_jitter"]]*rng.randn(num_qsos, num_bands)

    def perturb_hypers(self):
        k = rng.randint(num_hyperparameters)
        self.hypers[k] += ranges[k]*rng.randn()
        self.hypers[k] = wrap(self.hypers[k], bounds[k, 0], bounds[k, 1])

    def log_conditional_prior(self):
        pass
