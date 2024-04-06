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

class MonteCarloSimulation:
    """
    A class to simulate stock price paths with parameters for initial price,
    annual return, volatility and time.
    """

    def __init__(self,
                 mean_return: float,
                 volatility: float,
                 time: Optional[float] = None):
        """
        Initialize MonteCarloSimulation object with parameters for mean return,
        volatility and time. If no time is provided it defaults to 1 year.

        Parameters:
            mean_return (float): annual return rate of the stock
            volatility (float): standard deviation of price changes
            time (Optional[float] = None) -> Optional[float]: if not provided, default time is one year

        time: float = 1.0
        """
        self.mean_return = mean_return
        self.volatility = volatility
        self.time = time if time else 1.0

    def simulate(self, initial_price: float, simulations: int) -> np.ndarray:
        """
        Simulate stock price paths for a specified number of simulations and return the result as an array.

        Parameters:
            initial_price (float): initial price of the stock
            simulations (int): number of price path simulations to run

        Returns:
            np.ndarray: numpy array containing all simulated prices for each simulation
        """
        dt = 1 / self.time
        random_numbers = np.random.normal(0, dt ** 0.5,(simulations, int(self.time)))

        # Calculate price paths using the formula pt = pte * e^((r - sigma^2/2)t + sigma * dWt)
        prices = initial_price * np.exp(np.cumsum((self.mean_return - 0.5 * self.volatility ** 2) * dt \
                                                  + self.volatility * random_numbers, axis=1))

        return prices

# Execute function
def execute(mu: int, sigma: int, time_horizon: int, time_steps: int,
            initial_stock_price: int, num_simulations: int, random_seed: int) -> np.ndarray:
    # Initialize the Monte Carlo simulation with the specified parameters
    mc = MonteCarloSimulation(mu, sigma, time_horizon)
    
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

