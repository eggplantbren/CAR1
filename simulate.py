# Generate a simulated dataset

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rng

rng.seed(123)
a = 0.99
sigma = 3.0
b = sigma*np.sqrt(1.0 - a**2)

t = np.linspace(0.0, 1000.0, 20001)
y = np.zeros(t.shape)

y[0] = sigma*rng.randn()
for i in range(1, len(t)):
    y[i] = a*y[i-1] + b*rng.randn()
y += 5.0 + 0.1*rng.randn(len(t))

data = np.empty((len(t), 3))
data[:,0] = t
data[:,1] = y
data[:,2] = 0.1
np.savetxt("data.txt", data)

plt.errorbar(t, y, yerr=0.1, fmt="o")
plt.show()

