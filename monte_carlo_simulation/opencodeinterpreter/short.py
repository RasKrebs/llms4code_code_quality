import numpy as np
from scipy.stats import norm

# Define the parameters of the Monte Carlo simulation
expected_return = 0.1
volatility = 0.2
time_horizon = 10 # in years
time_steps = 100

def monte_carlo_simulation(expected_return, volatility, time_horizon, time_steps):
    # Calculate the number of steps per year and the total number of steps
    steps_per_year = time_steps / (time_horizon * 12)
    total_steps = int(steps_per_year * time_horizon * 12)

    # Initialize the price array with zeros
    prices = np.zeros(total_steps + 1)

    # Set the initial price
    prices[0] = 100

    # Generate the random numbers for the stock returns using a normal distribution
    returns = norm.rvs(loc=expected_return, scale=volatility, size=total_steps)

    # Iteratively calculate the prices based on the expected return and volatility
    for i in range(1, total_steps + 1):
        prices[i] = prices[i - 1] * (1 + returns[i - 1])

    # Return the simulated stock prices
    return prices
