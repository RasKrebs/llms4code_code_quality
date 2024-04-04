import numpy as np

def monte_carlo_simulation(S0, expected_return, volatility, time_horizon, time_steps):
    np.random.seed(42)
    
    dt = time_horizon / time_steps
    
    returns = np.exp((expected_return - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * np.random.randn(time_steps))
    
    prices = np.zeros(time_steps + 1)
    prices[0] = S0
    
    for t in range(1, time_steps + 1):
        prices[t] = prices[t-1] * returns[t-1]
        
    return prices
