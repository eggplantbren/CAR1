from extract_data import *
import celerite2
from celerite2 import terms
import numpy as np
import numpy.random as rng
from scipy.stats import norm, t

num_qsos = 190
num_bands = 3
num_hyperparameters = 12

data = []
for i in range(num_qsos):
    for band in ["g", "r", "i"]:
        data.append(get_data(i, band, sanitise=True, plot=False))
log10_lbol = np.array([d["lbol"] for d in data])
log10_lambda = np.array([d["log10_lambda"] for d in data])
z = np.array([d["redshift"] for d in data])
log10_1plusz = np.log10(1.0 + z)
mean_log10_lbol = np.mean(log10_lbol)
mean_log10_lambda = np.mean(log10_lambda)
mean_log10_1plusz = np.mean(log10_1plusz)

num_params = num_hyperparameters + 4*num_qsos*num_bands


names = ["beta0", "beta1", "beta2", "beta12",
         "n", "sig_log10_tau", "mu_mag", "sig_mag",
         "mu_log10_beta", "sig_log10_beta",
         "mu_log10_jitter", "sig_log10_jitter"]


def simulate_data():

    # A list for simulated data in the same format as the
    # real data
    simulated_data = []

    k = 0
    for i in range(num_qsos):
        for band in ["g", "r", "i"]:

            # Timestamps
            t = data[k]["light_curve"][:,0]
            n = len(t)

            # Generate a magnitude
            mu = 20.0 + 0.5*rng.randn()        

            # Generate a beta
            beta = 10.0**(-2.0 + 0.5*rng.randn())

            # Generate a tau
            beta0 = 3.5
            beta1 = 1.0
            beta2 = 1.0
            log10_tau = beta0 + beta1*(log10_lambda[k] - mean_log10_lambda) \
                           + beta2*(log10_lbol[k] - mean_log10_lbol) \
                           + 1.0*(log10_1plusz[k] - mean_log10_1plusz)
            tau = 10.0**log10_tau

            # Generate a jitter
            jitter = 10.0**(-2.0 + 1.0*rng.randn())

            # Compute sigma
            sigma = beta*np.sqrt(0.5*tau)

            # Construct covariance matrix
            delta = np.subtract.outer(t, t)
            C = sigma**2*np.exp(-np.abs(delta)/tau)
            C[np.diag_indices_from(C)] += jitter**2 + data[k]["light_curve"][:,2]**2
            L = np.linalg.cholesky(C)

            # Generate light curve
            y = mu + L @ rng.randn(n)
            array = np.column_stack((t, y, data[k]["light_curve"][:,2]))

            print(mu, beta, tau, jitter)
            
            simulated_data.append(dict(light_curve=array,
                                       redshift=data[k]["redshift"],
                                       log10_lambda=data[k]["log10_lambda"],
                                       lbol=data[k]["lbol"]))

            k += 1

    return simulated_data

if False:
    rng.seed(0)
    data = simulate_data()
    print("RUNNING ON FAKE DATA.")

if num_qsos < 190:
    print("NOT USING ALL QUASARS.")

def prior_transform(us):
    """
    Parameters are now:
    beta0, beta1, beta2, beta12, n, sigma from Brewer et al (2025) -
    these replace the log10_tau hyperparameters.
    Also mu, log10_beta, log10_jitter for each quasar and band
    and their hyperparameters.
    """

    beta0 = -10.0 + 20.0*us[0]
    beta1 = -10.0 + 20.0*us[1]
    beta2 = -10.0 + 20.0*us[2]
    beta12 = -1.0 + 2.0*us[3]
    n = -1.0 + 5.0*us[4]
    sig_log10_tau = 5.0*us[5]

    mu_mag = 0.0 + 40.0*us[6]
    sig_mag = 5.0*us[7]
    mu_log10_beta = -5.0 + 10.0*us[8]
    sig_log10_beta = 5.0*us[9]
    mu_log10_jitter = -2.0 # -5.0 + 10.0*us[10]
    sig_log10_jitter = 1.0 # 5.0*us[11]


    qso_params_3d = us[num_hyperparameters:].copy()
    qso_params_3d = qso_params_3d.reshape((num_qsos, num_bands, 4))

    qso_params_3d[:, :, 0] = norm.ppf(qso_params_3d[:, :, 0],
                                      loc=mu_mag, scale=sig_mag) # Mean magnitude
    qso_params_3d[:, :, 1] = norm.ppf(qso_params_3d[:, :, 1],
                                      loc=mu_log10_beta, scale=sig_log10_beta) # log10_beta in magnitudes

    # Regression line prediction for log10_tau
    reg = beta0 + beta1*(log10_lambda - mean_log10_lambda) \
                + beta2*(log10_lbol - mean_log10_lbol) \
                + beta12*(log10_lambda - mean_log10_lambda)*(log10_lbol - mean_log10_lbol) \
                + n*(log10_1plusz - mean_log10_1plusz)

    # Alternative for simple hierarchical version
    #reg = beta0 + 0*(log10_lambda - mean_log10_lambda)

    reg = reg.reshape((num_qsos, num_bands))

    qso_params_3d[:, :, 2] = norm.ppf(qso_params_3d[:, :, 2],
                                      loc=reg, scale=sig_log10_tau) # log10_tau in days
    qso_params_3d[:, :, 3] = norm.ppf(qso_params_3d[:, :, 3],
                                      loc=mu_log10_jitter, scale=sig_log10_jitter) # log10_jitter in magnitudes

    hypers = np.array([beta0, beta1, beta2, beta12, n, sig_log10_tau,
                       mu_mag, sig_mag, mu_log10_beta, sig_log10_beta,
                       mu_log10_jitter, sig_log10_jitter])

    return np.hstack([hypers, qso_params_3d.flatten()])

def log_likelihood(params):

    logl = 0.0

    qso_params_3d = params[num_hyperparameters:].reshape((num_qsos, num_bands, 4))

    mu =    qso_params_3d[:, :, 0]   
    beta =  10.0**qso_params_3d[:, :, 1]
    tau   = 10.0**qso_params_3d[:, :, 2]
    jitter= 10.0**qso_params_3d[:, :, 3]
    sigma = beta*np.sqrt(0.5*tau)

    k = 0
    for i in range(num_qsos):
        for j in range(num_bands):

            try:
                term = terms.RealTerm(a=sigma[i, j]**2, c=1.0/tau[i, j])
                kernel = term
                gp = celerite2.GaussianProcess(kernel, mean=mu[i, j])
                this_data = data[k]["light_curve"]
                gp.compute(this_data[:,0], yerr=np.sqrt(this_data[:,2]**2 + jitter[i, j]**2))
                logl += gp.log_likelihood(this_data[:,1])
            except Exception:
                logl += -1.0E300
            k += 1

    return logl


def both(us):
    return log_likelihood(prior_transform(us))

