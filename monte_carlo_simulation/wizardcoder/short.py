import numpy as np

def simulate_stock_price(S0, mu, sigma, T, dt):
    N = round(T/dt) # number of time steps
    t = np.linspace(0, T, N+1)  # discretized time array from 0 to T with step size dt
    S = np.zeros(N+1)  # stock price process

    # initial condition
    S[0] = S0

    for i in range(1, int(N + 1)):
        eps = np.random.normal()  # create random number from standard normal distribution
        S[i] = S[i - 1] * np.exp((mu - (sigma ** 2) / 2) * dt + sigma * np.sqrt(dt) * eps)

    return t, S
