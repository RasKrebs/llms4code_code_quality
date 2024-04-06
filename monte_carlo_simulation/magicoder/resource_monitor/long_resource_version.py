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

class StockMonteCarlo:
    """This is a Monte Carlo simulation class to simulate stock prices over time."""

    def __init__(self, expected_return: float, volatility: float,
                 time_horizon: int, time_steps: int) -> Optional[int]:
        self.expected_return = expected_return
        self.volatility = volatility
        if time_horizon <= 0 or time_steps <= 0:
            raise ValueError("Time horizon and steps must be positive integers.")

        self.time_horizon = float(time_horizon)
        self.time_steps = int(time_steps)
        self.delta_t = self.time_horizon / self.time_steps  # type: ignore

    def simulate(self, initial_stock_price: float, num_simulations: int, seed: Optional[int]=None) -> np.ndarray:
        """Generates the stock price paths using Monte Carlo simulation"""

        if initial_stock_price <= 0 or num_simulations <= 0:
            raise ValueError("Initial stock price and number of simulations must be positive integers.")

        # Generate random numbers with normal distribution.
        np.random.seed(seed)  # for reproducibility, set the seed

        dt = self.delta_t
        drift = (self.expected_return - 0.5 * self.volatility**2) * dt
        vol = self.volatility * np.sqrt(dt) * np.random.randn(num_simulations, self.time_steps)

        paths = np.empty((num_simulations, self.time_steps + 1))
        paths[:, 0] = initial_stock_price

        # Calculate each step of the path using Brownian motion
        for t in range(self.time_steps):
            paths[:, t+1] = paths[:, t] * np.exp(drift + vol[:, t])

        return paths

# Execute function
def execute(mu: int, sigma: int, time_horizon: int, time_steps: int,
            initial_stock_price: int, num_simulations: int, random_seed: int) -> np.ndarray:
    # Initialize the Monte Carlo simulation with the specified parameters
    mc = StockMonteCarlo(mu, sigma, time_horizon, time_steps)
    
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

