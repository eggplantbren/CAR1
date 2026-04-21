import numpy as np
import matplotlib.pyplot as plt
from mymodel import num_hyperparameters, names


logl = np.loadtxt("sample_info.txt")[:,1]
plt.plot(logl)
subset = logl[int(0.3*len(logl)):]
lower = np.min(subset)
upper = np.max(subset)
width = upper - lower
lower -= 0.05*width
upper += 0.05*width
plt.ylim([lower, upper])
plt.xlabel("Time")
plt.ylabel("Log Likelihood")
plt.show()

sample = np.loadtxt("sample.txt")
for i in range(num_hyperparameters):
    plt.plot(sample[:, i])
    plt.ylabel(names[i])
    plt.show()
