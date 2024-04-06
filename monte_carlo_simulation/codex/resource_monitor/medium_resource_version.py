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

class MonteCarlo:
    """
    A class used to simulate stock price paths using the Monte Carlo method.

    Attributes
    ----------
    annual_return : float
        The expected annual return of the stock.
    volatility : float
        The volatility of the stock.
    time : float
        The time horizon for the simulation.

    Methods
    -------
    simulate(initial_price: float, num_simulations: int) -> np.ndarray:
        Simulates the stock price paths.
    """

    def __init__(self, annual_return: float, volatility: float, time: float):
        """
        Parameters
        ----------
        annual_return : float
            The expected annual return of the stock.
        volatility : float
            The volatility of the stock.
        time : float
            The time horizon for the simulation.
        """
        self.annual_return = annual_return
        self.volatility = volatility
        self.time = time

    def simulate(self, initial_price: float, num_simulations: int) -> np.ndarray:
        """
        Simulates the stock price paths.

        Parameters
        ----------
        initial_price : float
            The initial price of the stock.
        num_simulations : int
            The number of simulations to run.

        Returns
        -------
        np.ndarray
            An array of simulated stock price paths.
        """
        # Calculate the daily return
        daily_return = self.annual_return / 252

        # Initialize the price paths array
        price_paths = np.zeros((252 * int(self.time), num_simulations))

        # Set the initial prices
        price_paths[0] = initial_price

        # Simulate the price paths
        for t in range(1, 252 * int(self.time)):
            brownian = np.random.normal(0, 1, num_simulations)
            price_paths[t] = price_paths[t - 1] * np.exp((daily_return - 0.5 * self.volatility**2) +
                                                         self.volatility * brownian)

        return price_paths

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

