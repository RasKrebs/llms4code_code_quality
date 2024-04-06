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



class MonteCarloSimulation:
    def __init__(self, annual_return: float, volatility: float, time_period: int):
        """
        Initialize the Monte Carlo Simulation with parameters for annual return,
        volatility, and time period.
        
        Parameters:
        annual_return (float): The average annual return of the stock
        volatility (float): The volatility of the stock
        time_period (int): The number of days to simulate
        """
        self.annual_return = annual_return
        self.volatility = volatility
        self.time_period = time_period
    
    def simulate(self, initial_price: float, num_simulations: int) -> np.array:
        """
        Simulate stock price paths using a Monte Carlo simulation.
        
        Parameters:
        initial_price (float): The initial price of the stock
        num_simulations (int): The number of simulations to run
        
        Returns:
        paths (np.array): An array of stock price paths (num_simulations x time_period)
        """
        paths = np.zeros((num_simulations, self.time_period))
        time = np.arange(1, self.time_period + 1)
        
        for i in range(num_simulations):
            path = np.zeros(self.time_period)
            path[0] = initial_price
            
            for t in range(1, self.time_period):
                daily_return = np.random.normal(self.annual_return/self.time_period, self.volatility)
                path[t] = path[t-1] * (1 + daily_return)
            
            paths[i, :] = path
        
        return paths

# Execute function
def execute(mu: int, sigma: int, time_horizon: int, time_steps: int,
            initial_stock_price: int, num_simulations: int, random_seed: int) -> np.ndarray:
    
    # Initialize the Monte Carlo simulation with the specified parameters
    mc = MonteCarloSimulation(mu, sigma, time_steps)
    
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

