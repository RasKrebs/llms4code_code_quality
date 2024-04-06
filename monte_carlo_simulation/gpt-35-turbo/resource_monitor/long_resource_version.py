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
import numpy as np
import numpy.random as npr

class MonteCarlo:
    def __init__(self, expected_return: float, volatility: float, time_horizon: int, time_steps: int):
        """
        Initialize the MonteCarlo simulation with parameters.

        Parameters:
        expected_return (float): The expected return of the stock.
        volatility (float): The volatility of the stock.
        time_horizon (int): The total time duration of the simulation.
        time_steps (int): The number of time steps in the simulation.
        """
        self.expected_return = expected_return
        self.volatility = volatility
        self.time_horizon = time_horizon
        self.time_steps = time_steps
        self.time_step_size = time_horizon / time_steps

        # Validate input parameters
        if self.time_step_size <= 0:
            raise ValueError("Time step size must be greater than 0.")
    
    def simulate(self, initial_stock_price: float, num_simulations: int, random_seed: int = None) -> np.ndarray:
        """
        Generate stock price paths through Monte Carlo simulation.

        Parameters:
        initial_stock_price (float): The initial stock price for simulation.
        num_simulations (int): The number of simulation paths to generate.
        random_seed (int): The random seed for reproducibility.

        Returns:
        np.ndarray: A numpy array of stock price paths with shape (num_simulations, time_steps+1).
        """
        if num_simulations <= 0:
            raise ValueError("Number of simulations must be greater than 0.")
        
        # Set random seed for reproducibility
        if random_seed:
            npr.seed(random_seed)
        
        stock_paths = np.zeros((num_simulations, self.time_steps + 1))
        stock_paths[:, 0] = initial_stock_price

        for t in range(1, self.time_steps + 1):
            # Generate random samples from a normal distribution
            rand_samples = npr.normal(self.expected_return * self.time_step_size, 
                                      self.volatility * np.sqrt(self.time_step_size), 
                                      num_simulations)
            stock_paths[:, t] = stock_paths[:, t - 1] * np.exp(rand_samples)
        
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

