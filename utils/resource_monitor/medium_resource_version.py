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
from typing import List, Tuple

class MonteCarlo:
    """
    This class is created to simulate stock price paths based on the Black-Scholes model.
    It includes attributes for annual return, volatility, and time parameters
    along with a simulate function to generate price paths.
    """

    def __init__(self, annual_return: float, volatility: float, time: int):
        self.annual_return = annual_return
        self.volatility = volatility
        self.time = time

    def simulate(self, initial_price: float, simulation_count: int) -> List[List[float]]:
        """
        Simulates the stock price paths based on the Black-Scholes model.

        Parameters
        ----------
        initial_price : float
            The starting stock price.
        simulation_count : int
            The number of simulations to run.

        Returns
        -------
        List[List[float]]
            A list containing the simulated stock price paths for each time step.

        Examples
        --------
        >>> mc = MonteCarlo(0.05, 0.2, 1)
        >>> paths = mc.simulate(100, 1000)
        >>> print(paths[0])  # The first path will have the initial price followed by simulated prices for each time step.
        [100, ..., ...]
        """
        dt = 1/365  # Daily return
        paths = []  # List to store all the simulations

        # Simulating each path
        for _ in range(simulation_count):
            price_path = [initial_price]  # Starting with initial stock price

            # Generating each time step
            for t in range(1, self.time + 1):
                drift = (self.annual_return - 0.5 * self.volatility**2) * dt
                shock = np.random.normal(loc=0, scale=self.volatility*np.sqrt(dt))

                price = price_path[-1] * np.exp(drift + shock)
                price_path.append(price)  # Appending the new stock price to the path

            paths.append(price_path)

        return paths

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

