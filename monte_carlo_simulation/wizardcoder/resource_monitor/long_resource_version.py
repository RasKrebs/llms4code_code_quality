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
from typing import Optional

class MonteCarlo:
    def __init__(self, expected_return: float, volatility: float, time_horizon: float, time_steps: int):
        """
        Initialize a Monte Carlo object with the specified parameters.

        :param expected_return: The average return of the stock (annualized)
        :param volatility: The standard deviation of daily returns (annualized)
        :param time_horizon: Total length of simulation, in years
        :param time_steps: Number of steps to use for the simulation

        """
        if expected_return < 0 or volatility < 0 or time_horizon <= 0 or time_steps <= 0:
            raise ValueError("Parameters must be positive.")

        self.expected_return = expected_return
        self.volatility = volatility
        self.time_horizon = time_horizon
        self.time_steps = time_steps
        self.time_step_size = time_horizon / float(time_steps)  # Calculate time step size

    def simulate(self, initial_price: float, num_simulations: int, seed: Optional[int] = None):
        """
        Run a Monte Carlo simulation to generate stock price paths.

        :param initial_price: The starting price of the stock at time 0
        :param num_simulations: Number of simulations to run
        :param seed: An optional random number generator seed for reproducibility
        :return: A NumPy array of simulated stock prices, with shape (num_steps + 1, num_simulations)

        """
        if initial_price <= 0 or num_simulations <= 0:
            raise ValueError("Parameters must be positive.")

        try:
            # Set the seed for the random number generator
            np.random.seed(seed)

            # Define the mean and standard deviation of daily returns
            mu = self.expected_return / self.time_steps
            sigma = self.volatility / np.sqrt(self.time_steps)

            # Generate random daily returns for each time step and simulation
            rand_returns = np.random.normal(loc=mu, scale=sigma, size=(self.time_steps, num_simulations))

            # Calculate the cumulative product to get the paths
            price_paths = initial_price * np.cumprod(1 + rand_returns, axis=0)

            return price_paths
        except Exception as e:
            print("An error occurred during simulation.")
            raise e

# Execute function
def execute(mu: int, sigma: int, time_horizon: int, time_steps: int,
            initial_stock_price: int, num_simulations: int, random_seed: int) -> np.ndarray:
    # Initialize the Monte Carlo simulation with the specified parameters
    mc = MonteCarlo(mu, sigma, time_horizon, time_steps)
    
    # Perform the simulation
    simulated_stock_paths = mc.simulate(initial_stock_price, num_simulations, random_seed)
    
    return simulated_stock_paths



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

