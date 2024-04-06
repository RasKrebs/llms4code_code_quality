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
from typing import Tuple, List

class MonteCarlo:
    def __init__(self, expected_return: float, volatility: float, time_horizon: int, time_steps: int) -> None:
        self.expected_return = expected_return
        self.volatility = volatility
        self.time_horizon = time_horizon
        self.time_steps = time_steps
        self.time_step_size = self.time_horizon / self.time_steps  # Calculate the size of each time step

    def simulate(self, initial_price: float, num_simulations: int, seed: int) -> np.ndarray:
        try:
            # Seed the random number generator for reproducibility
            np.random.seed(seed)

            stock_paths = []  # List to store all simulated paths

            # For each simulation
            for _ in range(num_simulations):
                path = [initial_price]  # Initialize the path with initial price

                # Simulate each time step
                for _ in range(self.time_steps):
                    random_number = np.random.normal()  # Generate a random number from standard normal distribution

                    # Update the next stock price based on the expected return, volatility and random number
                    next_price = path[-1] * (1 + self.expected_return * self.time_step_size + self.volatility * np.sqrt(self.time_step_size) * random_number)

                    # Append the new stock price to the path
                    path.append(next_price)

                # Append the completed path to the list of paths
                stock_paths.append(path)

            return np.array(stock_paths)  # Convert the list of paths into a numpy array and return it

        except Exception as e:
            raise ValueError('An error occurred during simulation: ' + str(e)) from e

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

