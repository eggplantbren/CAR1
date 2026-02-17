import numpy as np
import matplotlib.pyplot as plt


logl = np.loadtxt("sample_info.txt")[:,1]
plt.plot(logl)
lower = np.sort(logl)[int(0.05*len(logl))]
upper = logl.max()
upper += 0.05*(upper - lower)
plt.ylim([lower, upper])
plt.xlabel("Time")
plt.ylabel("Log Likelihood")
plt.show()
