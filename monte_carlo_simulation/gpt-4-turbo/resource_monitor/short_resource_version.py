import psutil
import os
import threading
import multiprocessing

# Define a global variable to store the maximum resources usage
max_resources_usage = {"cpu": 0, "memory": 0}


import numpy as np
np.random.seed(42)

# Load data/parameters section
mu = 0.1  # Expected annual return
sigma = 0.2  # Volatility
time_horizon = 2  # Time horizon in years
time_steps = 252  # Number of time steps, assuming 252 trading days in a year
initial_stock_price = 100  # Initial stock price
num_simulations = 10000  # Number of simulations
random_seed = 42  # Random seed for reproducibility


# Implement the resource monitor
def resource_monitor():
    """
    Monitors the CPU and memory usage of the current process, updating global max usage.
    """
    global max_resources_usage
    process = psutil.Process(os.getpid())
    
    while monitoring:
        cpu_usage = process.cpu_percent(interval=1) / multiprocessing.cpu_count()
        memory_usage = process.memory_info().rss
        max_resources_usage['cpu'] = max(max_resources_usage['cpu'], cpu_usage)
        max_resources_usage['memory'] = max(max_resources_usage['memory'], memory_usage)



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
    num_steps = int(dt/T)
    
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

# Execute function
def execute(mu: int, sigma: int, time_horizon: int, time_steps: int,
            initial_stock_price: int, num_simulations: int, random_seed: int) -> np.ndarray:
    # Initialize the Monte Carlo simulation with the specified parameters
    mc = simulate_stock_prices(initial_stock_price, mu, sigma, time_horizon, 252, num_simulations)
    
    return mc



if __name__ == "__main__":
    # Start the resource monitoring in a separate thread
    global monitoring
    monitoring = True
    monitor_thread = threading.Thread(target=resource_monitor)
    monitor_thread.start()

    # Execute the Huffman coding process

    # Using the execute function
    simulated_stock_paths = execute(mu, sigma, time_horizon, time_steps, initial_stock_price, num_simulations, random_seed)


    # Stop the monitoring
    monitoring = False
    monitor_thread.join()

    print(max_resources_usage['cpu']), print(max_resources_usage['memory'] / (1024**2))

