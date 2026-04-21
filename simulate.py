from mymodel import *
import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt

# A list for simulated data in the same format as the
# real data
simulated_data = []


for i in range(num_qsos):
    for band in ["g", "r", "i"]:

        # Timestamps
        t = data[i]["light_curve"][:,0]
        n = len(t)

        # Generate a magnitude
        mu = 20.0 + 0.5*rng.randn()        

        # Generate a beta
        beta = 10.0**(-2.0 + 0.5*rng.randn())

        # Generate a tau
        tau = 10.0**(3.5 + 0.5*rng.randn())

        # Generate a jitter
        jitter = 10.0**(-2.0 + 1.0*rng.randn())

        # Compute sigma
        sigma = beta*np.sqrt(0.5*tau)

        # Construct covariance matrix
        rows, cols = np.indices((n, n))
        delta = np.subtract.outer(t, t)
        C = sigma*np.exp(-np.abs(delta)/tau)
        L = np.linalg.cholesky(C)

        print(C)
        plt.imshow(C)
        plt.show()
