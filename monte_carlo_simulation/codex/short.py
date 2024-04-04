import numpy as np

def monte_carlo_simulation(initial_price, expected_return, volatility, time_horizon, time_steps):
    # Calculate the time step size
    dt = time_horizon / time_steps

    # Generate random numbers for the simulation
    random_numbers = np.random.normal(0, 1, time_steps)

    # Initialize an array to store the price at each time step
    prices = np.zeros(time_steps)
    prices[0] = initial_price

    # Perform the simulation
    for t in range(1, time_steps):
        prices[t] = prices[t - 1] * np.exp((expected_return - 0.5 * volatility**2) * dt 
                                            + volatility * np.sqrt(dt) * random_numbers[t])

    return prices