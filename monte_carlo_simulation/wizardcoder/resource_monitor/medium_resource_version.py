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
from typing import List

class MonteCarloStockPrice:
    def __init__(self, annual_return: float, volatility: float, time: int) -> None:
        """
        Initialize the class with parameters.

        Parameters:
            annual_return (float): The expected return of the stock annually.
            volatility (float): The standard deviation of log returns.
            time (int): The total number of periods for which to simulate prices.
        """
        self.annual_return = annual_return
        self.volatility = volatility
        self.time = time

    def simulate(self, initial_price: float, num_simulations: int) -> List[List[float]]:
        """
        Generate a number of price paths based on the Monte Carlo simulation methodology.

        Parameters:
            initial_price (float): The starting price for all simulations.
            num_simulations (int): The total number of simulations to generate.

        Returns:
            A list of lists, where each inner list represents a single simulated path of prices.
        """
        # Define the mean and standard deviation for the normal distribution used in the Monte Carlo simulation
        mu = self.annual_return - (self.volatility ** 2) / 2
        sigma = self.volatility

        dt = 1.0/365  # daily returns
        total_time = 1.0 * self.time

        paths = []
        for _ in range(num_simulations):
            path = [initial_price]

            for _ in np.arange(dt, total_time + dt, dt):  # generate prices for each time step
                rtn = np.random.normal(mu * dt, sigma * np.sqrt(dt))  # daily returns
                price = path[-1] * np.exp(rtn)  # new price
                path.append(price)

            paths.append(path)

        return paths

# Execute function
def execute(mu: float, sigma: float, time_horizon: int, time_steps: int,
            initial_stock_price: float, num_simulations: int, random_seed: int) -> List[List[float]]:
    # Initialize the Monte Carlo simulation with the specified parameters
    mc = MonteCarloStockPrice(mu, sigma, time_horizon)
    
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

