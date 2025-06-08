# Generate a simulated dataset

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rng

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 14,
})

rng.seed(12)
a = 0.99999
sigma = 10.0
b = sigma*np.sqrt(1.0 - a**2)

# Must use an arange for this - I am assuming that later.
t = np.arange(10000)
y = np.empty(t.shape)

y[0] = sigma*rng.randn()
for i in range(1, len(t)):
    y[i] = a*y[i-1] + b*rng.randn()

# Observed points
t_obs = 4000 + rng.randint(1000, size=100)
y_obs = y[t_obs]
y_obs += 0.1*rng.randn()

# Plot the curve
plt.plot(t, y, "r-", alpha=0.5, label="Underlying curve")

# Plot the data points
plt.errorbar(t_obs, y_obs, yerr=0.1, fmt=".", label="Observations")

plt.legend()
plt.xlabel("Time $t$")
plt.ylabel("Signal $y(t)$")

# Save and display the plot
plt.savefig("simulation.pdf")
plt.show()

