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
    """
    A class to simulate stock price paths using the Monte Carlo method.

    Attributes:
        expected_return (float): The expected return of the stock.
        volatility (float): The volatility of the stock.
        time_horizon (float): The time horizon for the simulation.
        time_steps (int): The number of time steps for the simulation.
        time_step_size (float): The size of each time step.
    """

    def __init__(self, expected_return: float, volatility: float, time_horizon: float, time_steps: int):
        """
        The constructor for the MonteCarlo class.

        Args:
            expected_return (float): The expected return of the stock.
            volatility (float): The volatility of the stock.
            time_horizon (float): The time horizon for the simulation.
            time_steps (int): The number of time steps for the simulation.
        """

        # Validate inputs
        if expected_return < 0:
            raise ValueError("Expected return must be non-negative.")
        if volatility < 0:
            raise ValueError("Volatility must be non-negative.")
        if time_horizon <= 0:
            raise ValueError("Time horizon must be positive.")
        if time_steps <= 0:
            raise ValueError("Time steps must be positive.")

        self.expected_return = expected_return
        self.volatility = volatility
        self.time_horizon = time_horizon
        self.time_steps = time_steps
        self.time_step_size = time_horizon / time_steps

    def simulate(self, initial_price: float, num_simulations: int, random_seed: int = None) -> np.ndarray:
        """
        Simulate stock price paths using the Monte Carlo method.

        Args:
            initial_price (float): The initial price of the stock.
            num_simulations (int): The number of simulations to run.
            random_seed (int, optional): The random seed for reproducibility. Defaults to None.

        Returns:
            np.ndarray: A numpy array of stock price paths.
        """

        # Validate inputs
        if initial_price <= 0:
            raise ValueError("Initial price must be positive.")
        if num_simulations <= 0:
            raise ValueError("Number of simulations must be positive.")

        # Set the random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)

        # Initialize the price paths array
        price_paths = np.zeros((num_simulations, self.time_steps + 1))
        price_paths[:, 0] = initial_price

        # Generate the price paths
        for t in range(1, self.time_steps + 1):
            # Generate random returns
            returns = np.random.normal(self.expected_return, self.volatility, num_simulations)

            # Update the price paths
            price_paths[:, t] = price_paths[:, t - 1] * np.exp(returns)

        return price_paths


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

