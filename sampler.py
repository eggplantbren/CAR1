# A custom metropolis sampler for the hierarchical model.
import numpy as np
import numpy.random as rng

def wrap(x, a, b):
    L = b - a
    return a + (x - a) - L * np.floor((x - a) / L)

def randh():
    return 10.0**(1.5 - 6.0*rng.rand())*rng.randn()

num_hyperparameters = 8
num_quasars = 190
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

        # quasar parameters
        self.mu = self.hypers[h_idx["mu_mu"]] + self.hypers[h_idx["sig_mu"]]*rng.randn(num_quasars, num_bands)
        self.log10_beta = self.hypers[h_idx["mu_log10_beta"]] + self.hypers[h_idx["sig_log10_beta"]]*rng.randn(num_quasars, num_bands)
        self.log10_tau = self.hypers[h_idx["mu_log10_tau"]] + self.hypers[h_idx["sig_log10_tau"]]*rng.randn(num_quasars, num_bands)
        self.log10_jitter = self.hypers[h_idx["mu_log10_jitter"]] + self.hypers[h_idx["sig_log10_jitter"]]*rng.randn(num_quasars, num_bands)

    def perturb_hypers(self):
        k = rng.randint(num_hyperparameters)
        self.hypers[k] += ranges[k]*rng.randn()
        self.hypers[k] = wrap(self.hypers[k], bounds[k, 0], bounds[k, 1])
        return 0.0

    def perturb_quasar(self):
        quasar = rng.randint(num_quasars)
        band = rng.randint(num_bands)

        logh = 0.0
        logh -= -0.5*((self.mu[quasar, band] - self.hypers[h_idx["mu_mu"]])/self.hypers[h_idx["sig_mu"]])**2
        logh -= -0.5*((self.log10_beta[quasar, band] - self.hypers[h_idx["mu_log10_beta"]])/self.hypers[h_idx["sig_log10_beta"]])**2
        logh -= -0.5*((self.log10_tau[quasar, band] - self.hypers[h_idx["mu_log10_tau"]])/self.hypers[h_idx["sig_log10_tau"]])**2
        logh -= -0.5*((self.log10_jitter[quasar, band] - self.hypers[h_idx["mu_log10_jitter"]])/self.hypers[h_idx["sig_log10_jitter"]])**2
        self.mu[quasar, band] += self.hypers[h_idx["sig_mu"]]*randh()
        self.log10_beta[quasar, band] += self.hypers[h_idx["sig_log10_beta"]]*randh()
        self.log10_tau[quasar, band] += self.hypers[h_idx["sig_log10_tau"]]*randh()
        self.log10_jitter[quasar, band] += self.hypers[h_idx["sig_log10_jitter"]]*randh()
        logh += -0.5*((self.mu[quasar, band] - self.hypers[h_idx["mu_mu"]])/self.hypers[h_idx["sig_mu"]])**2
        logh += -0.5*((self.log10_beta[quasar, band] - self.hypers[h_idx["mu_log10_beta"]])/self.hypers[h_idx["sig_log10_beta"]])**2
        logh += -0.5*((self.log10_tau[quasar, band] - self.hypers[h_idx["mu_log10_tau"]])/self.hypers[h_idx["sig_log10_tau"]])**2
        logh += -0.5*((self.log10_jitter[quasar, band] - self.hypers[h_idx["mu_log10_jitter"]])/self.hypers[h_idx["sig_log10_jitter"]])**2

        return logh


    def __str__(self):
        s = ""
        for i in range(num_hyperparameters):
            s += str(self.hypers[i]) + " "

        flattened = np.hstack([self.mu.flatten(),
                               self.log10_beta.flatten(),
                               self.log10_tau.flatten(),
                               self.log10_jitter.flatten()])
        for x in flattened:
            s += str(x) + " "
                
        return s


if __name__ == "__main__":
    import copy

    particle = Particle()

    for i in range(100000):
        proposal = copy.deepcopy(particle)

        if rng.rand() <= 0.9:
            logh = proposal.perturb_quasar()
        else:
            logh = proposal.perturb_hypers()

        if rng.rand() <= np.exp(logh):
            particle = proposal

        if (i+1)%100 == 0:
            print(i+1, particle)

