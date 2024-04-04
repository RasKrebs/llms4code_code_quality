import numpy as np
import matplotlib.pyplot as plt

def simulate_stock_prices(S0, expected_return, volatility, T, dt, num_simulations):
    """
    Simulates stock prices using Geometric Brownian Motion.

    Parameters:
    S0 (float): Initial stock price.
    expected_return (float): The expected return of the stock (mu).
    volatility (float): The volatility of the stock returns (sigma).
    T (float): Time horizon in years.
    dt (float): Time step in years.
    num_simulations (int): The number of simulation paths.

    Returns:
    numpy.ndarray: Simulated stock price paths.
    """
    # Calculate the number of time steps
    num_steps = int(T / dt)
    
    # Generate random increments for the Brownian motion
    # Using np.random.normal to generate random numbers with a normal distribution
    random_increments = np.random.normal(0, 1, (num_steps, num_simulations))
    
    # Calculate the drift and volatility terms
    drift = (expected_return - 0.5 * volatility ** 2) * dt
    diffusion = volatility * np.sqrt(dt) * random_increments
    
    # Combine drift and diffusion terms
    stock_price_changes = drift + diffusion
    
    # Create an array to hold the simulated stock prices
    stock_prices = np.zeros_like(stock_price_changes)
    stock_prices[0] = S0
    
    # Compute the simulated stock prices recursively
    for t in range(1, num_steps):
        stock_prices[t] = stock_prices[t-1] * np.exp(stock_price_changes[t])
    
    return stock_prices
