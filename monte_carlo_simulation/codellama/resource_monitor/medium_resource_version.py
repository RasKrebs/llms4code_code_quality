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
from typing import Tuple

class MonteCarlo:
    """A class for simulating stock price paths using Monte Carlo simulation."""

    def __init__(self, mu: float, sigma: float, T: int):
        self.mu = mu
        self.sigma = sigma
        self.T = T

    def simulate(self, initial_price: float, simulation_count: int) -> Tuple[np.ndarray, np.ndarray]:
        """Run Monte Carlo simulations to generate stock price paths.

        Args:
            initial_price (float): Initial price of the underlying asset.
            simulation_count (int): Number of simulations to run.

        Returns:
            time_steps (np.ndarray): Array of time steps for each simulation.
            paths (np.ndarray): Simulated price paths as a 2-D array, with rows representing individual paths and columns representing time steps.
        """

        # Set up the parameters
        dt = 1 / self.T
        count = np.arange(simulation_count)

        # Reshape time_steps to match the shape of Z
        time_steps = dt * np.arange(self.T + 1)
        time_steps = np.reshape(time_steps, (self.T + 1, 1))

        # Generate standard normal random numbers for simulating price paths
        Z = np.random.standard_normal((self.T + 1, simulation_count))

        # Simulate the price paths
        paths = initial_price * np.exp((self.mu - self.sigma**2 / 2) * time_steps + self.sigma * np.sqrt(dt) * Z)

# Execute function
def execute(mu: int, sigma: int, time_horizon: int, time_steps: int,
            initial_stock_price: int, num_simulations: int, random_seed: int) -> np.ndarray:
    # Initialize the Monte Carlo simulation with the specified parameters
    mc = MonteCarlo(mu, sigma, time_horizon)
    
    # Perform the simulation
    simulated_stock_paths = mc.simulate(initial_stock_price, num_simulations)
    
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

