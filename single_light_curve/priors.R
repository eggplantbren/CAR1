
n = 1E5
log10_sigma = rnorm(n, mean=0, sd=1)
log10_beta  = rnorm(n, mean=-2, sd=0.5)
log10_tau = (log10_sigma - log10_beta + 0.5*log10(2))*2

hist(log10_tau, breaks=100)
