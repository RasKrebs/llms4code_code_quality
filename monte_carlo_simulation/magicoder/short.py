import numpy as np
from scipy.stats import norm

def monte_carlo_simulation(S0, mu, sigma, T, dt, n):
    N = int(T/dt) # number of time steps

    S = np.zeros((N+1,n)) # stock price matrix
    R = norm.rvs(size=(N,n),scale=np.sqrt(dt)*sigma) # random numbers

    S[0] = S0 # initial stock prices

    for t in range(1, N+1):
        S[t] = S[t-1]*np.exp((mu - 0.5 * sigma**2)*(dt) + sigma * R[t-1])

    return S[-1][:] # return final stock prices
