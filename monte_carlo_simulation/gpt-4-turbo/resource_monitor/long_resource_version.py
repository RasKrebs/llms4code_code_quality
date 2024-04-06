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
    This class performs Monte Carlo simulations for stock price forecasts
    using the geometric Brownian motion model.
    
    Attributes:
        expected_return (float): The expected return of the stock.
        volatility (float): The volatility of the stock.
        time_horizon (float): The time horizon for the simulation in years.
        time_steps (int): The number of time steps in the simulation.
        time_step_size (float): The size of each time step, computed at initialization.
    """
    
    def __init__(self, expected_return: float, volatility: float, 
                 time_horizon: float, time_steps: int) -> None:
        """
        Initializes the MonteCarlo simulator with the specified parameters.
        
        Args:
            expected_return (float): The expected return of the stock.
            volatility (float): The volatility of the stock.
            time_horizon (float): The time horizon for the simulation in years.
            time_steps (int): The number of time steps in the simulation.
        
        Raises:
            ValueError: If any of the parameter values are not positive.
        """
        if expected_return <= 0 or volatility <= 0 or time_horizon <= 0 or time_steps <= 0:
            raise ValueError("All parameters must be positive values.")
        
        self.expected_return = expected_return
        self.volatility = volatility
        self.time_horizon = time_horizon
        self.time_steps = time_steps
        self.time_step_size = self.time_horizon / self.time_steps
        
    def simulate(self, initial_stock_price: float, num_simulations: int, random_seed: int = None) -> np.ndarray:
        """
        Simulates the stock price paths using the Monte Carlo method.

        Args:
            initial_stock_price (float): The initial stock price.
            num_simulations (int): The number of simulations to perform.
            random_seed (int, optional): A seed for the random number generator
                                         for reproducibility. Defaults to None.

        Returns:
            np.ndarray: An array of shape (num_simulations, time_steps + 1)
                        containing the stock price paths.
                        
        Raises:
            ValueError: If the initial_stock_price is not positive or
                        if num_simulations is not a positive integer.
        """
        if initial_stock_price <= 0:
            raise ValueError("The initial stock price must be a positive value.")
        if num_simulations <= 0:
            raise ValueError("The number of simulations must be a positive integer.")
        
        np.random.seed(random_seed)  # Set the random seed for reproducibility
        
        # Pre-compute constants for the simulation formula
        drift = self.expected_return - (0.5 * self.volatility ** 2)
        shock = self.volatility * np.sqrt(self.time_step_size)
        
        # Initialize the array to hold the simulation results
        stock_paths = np.zeros((num_simulations, self.time_steps + 1))
        stock_paths[:, 0] = initial_stock_price  # Set initial prices
        
        # Perform the simulations
        for t in range(1, self.time_steps + 1):
            # Random increments: Brownian motion
            rand_increments = np.random.normal(0, 1, num_simulations)
            # Calculate the stock price for this time step
            stock_paths[:, t] = stock_paths[:, t-1] * np.exp(drift * self.time_step_size + shock * rand_increments)
        
        return stock_paths

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

