# Generate a simulated dataset

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rng

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 12,
})

days = 365*100
window = np.arange(15000, 15000+20*365)
err = 0.015
n = 250

def generate_light_curve(save=True, plot=True):
    mu = 20.0 + 3.0*rng.randn()
    log10_beta = -2.0 + 0.3*rng.randn()
    log10_tau = 4.0 + 2.0*rng.randn()
    log10_jitter = -2.0 + 0.5*rng.randn()
    beta, tau, jitter = 10.0**log10_beta, 10.0**log10_tau, 10.0**log10_jitter
    alpha = np.exp(-1.0/tau)
    sigma = beta*np.sqrt(0.5*tau)
    eps = sigma*np.sqrt(1.0 - alpha**2)

    y = np.empty(days)
    y[0] = mu + sigma*rng.randn()
    for i in range(1, days):
        y[i] = mu + alpha*(y[i-1] - mu) + eps*rng.randn()

    t = np.arange(days)
    choice = rng.choice(window, size=n, replace=False)
    choice = np.sort(choice)
    t_obs = t[choice]
    y_obs = y[choice] + np.sqrt(jitter**2 + err**2)*rng.randn(n)
    err_obs = err*np.ones(n)

    # Save the data
    data = np.column_stack((t_obs, y_obs, err_obs))
    if save:
        np.savetxt("data.txt", data)

    if plot:
        # Two panels in the plot
        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.errorbar(t_obs, y_obs, yerr=err_obs,
                      fmt=".", label="Observations")

        plt.subplot(2, 1, 2)
        plt.plot(t, y, "r-", alpha=0.5, label="Underlying curve")
        plt.errorbar(t_obs, y_obs, yerr=err_obs,
                     fmt=".", label="Observations")

        plt.legend()
        plt.xlabel("Time $t$ (days)")
        plt.ylabel("Signal $y(t)$ (magnitudes)")

        # Save and display the plot
        plt.savefig("simulation.pdf", bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    rng.seed(123)
    generate_light_curve()
