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
from scipy.stats import norm

class MonteCarlo:
    def __init__(self, expected_return: float, volatility: float, time_horizon: int, time_steps: int):
        """
        Initializes the Monte Carlo simulation parameters.

        Args:
            expected_return (float): Expected return of the asset per unit time
            volatility (float): Volatility of the asset's returns
            time_horizon (int): Time horizon for the simulation in years
            time_steps (int): Number of steps to discretize the simulation into
        """
        if expected_return < 0 or volatility < 0:
            raise ValueError("Expected return and volatility must be positive.")

        self.expected_return = expected_return
        self.volatility = volatility
        self.time_horizon = time_horizon
        self.time_steps = time_steps
        self.time_step_size = time_horizon / time_steps  # Size of each time step in years

    def simulate(self, initial_price: float, num_simulations: int, seed: int) -> np.ndarray:
        """
        Simulates stock price paths using a Brownian motion process (also known as a Wiener process).

        Args:
            initial_price (float): Initial price of the asset at time 0
            num_simulations (int): Number of simulations to generate
            seed (int): Random seed for reproducibility

        Returns:
            np.ndarray: A numpy array of shape (num_simulations, time_steps + 1) containing the simulated stock price paths.
        """

        # Initialize the simulation matrix with zeros
        simulations = np.zeros((num_simulations, self.time_steps + 1))

        # Set the initial prices
        simulations[:, 0] = initial_price

        # Generate random standard normal variates for the Brownian motion process
        np.random.seed(seed)
        brownian_motion = norm.rvs(size=(num_simulations, self.time_steps))

        for i in range(1, self.time_steps + 1):
            # Simulate the stock price at time step i using a Brownian motion process:
            # S[i] = S[i-1] * exp((r - 0.5*sigma^2)*dt + sigma * sqrt(dt) * Z[i])

            simulations[:, i] = (
                simulations[:, i-1] * np.exp(
                    (self.expected_return - 0.5 * self.volatility ** 2) * self.time_step_size +
                    self.volatility * np.sqrt(self.time_step_size) * brownian_motion[:, i-1]
                )
            )

        return simulations

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

