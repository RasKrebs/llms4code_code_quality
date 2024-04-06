import psutil
import os
import threading
import multiprocessing

# Define a global variable to store the maximum resources usage
max_resources_usage = {"cpu": 0, "memory": 0}


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

# Execute function
def execute(mu: float, sigma: float, time_horizon: int, time_steps: int,
            initial_stock_price: float, num_simulations: int, random_seed: int) -> np.ndarray:
    # Define the parameters of the Monte Carlo simulation
    expected_return = mu
    volatility = sigma
    
    # Calculate the number of steps per year and the total number of steps
    steps_per_year = time_steps / (time_horizon * 12)
    total_steps = int(steps_per_year * time_horizon * 12)
    
    # Initialize the price array with zeros
    prices = np.zeros(total_steps + 1)
    
    # Set the initial price
    prices[0] = initial_stock_price
    
    # Generate the random numbers for the stock returns using a normal distribution
    returns = norm.rvs(loc=expected_return, scale=volatility, size=total_steps)
    
    # Iteratively calculate the prices based on the expected return and volatility
    for i in range(1, total_steps + 1):
        prices[i] = prices[i - 1] * (1 + returns[i - 1])
    
    # Return the simulated stock prices
    return prices



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

