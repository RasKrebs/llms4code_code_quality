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

class MonteCarloStockSimulator:
    """
    A class to simulate stock price paths using the Monte Carlo method.
    
    Attributes:
        annual_return (float): The expected annual return of the stock.
        volatility (float): The annual volatility of the stock's returns.
        time_horizon (int): The time horizon for the simulation in years.
        
    Methods:
        simulate(initial_price: float, n_simulations: int) -> np.ndarray:
            Generates simulated stock price paths.
    """
    
    def __init__(self, annual_return: float, volatility: float, time_horizon: int):
        """
        Initializes the MonteCarloStockSimulator with expected annual return, 
        volatility, and time horizon for the simulation.
        
        Parameters:
            annual_return (float): The expected annual return of the stock.
            volatility (float): The annual volatility of the stock returns.
            time_horizon (int): The time horizon for the simulation in years.
        """
        self.annual_return = annual_return
        self.volatility = volatility
        self.time_horizon = time_horizon
        
    def simulate(self, initial_price: float, n_simulations: int) -> np.ndarray:
        """
        Simulates stock price paths over the specified time horizon.
        
        Using the Geometric Brownian Motion model for stock price evolution,
        this function generates several simulated paths for the stock price, 
        based on its annual return, volatility, and a given initial price.
        
        Parameters:
            initial_price (float): The starting price of the stock.
            n_simulations (int): The number of simulated paths to generate.
        
        Returns:
            np.ndarray: A 2D array where each row represents a simulated stock price path.
        """
        # Daily returns are used for simulation
        days = self.time_horizon * 252  # assuming 252 trading days in a year
        daily_return = self.annual_return / 252
        daily_volatility = self.volatility / np.sqrt(252)
        
        # Random component in Geometric Brownian Motion
        random_shocks = np.random.normal(loc=daily_return, scale=daily_volatility, size=(days, n_simulations))
        
        # Calculating price paths
        price_paths = np.zeros_like(random_shocks)
        price_paths[0] = initial_price
        
        for t in range(1, days):
            price_paths[t] = price_paths[t-1] * np.exp(random_shocks[t])
        
        return price_paths

# Execute function
def execute(mu: int, sigma: int, time_horizon: int, time_steps: int,
            initial_stock_price: int, num_simulations: int, random_seed: int) -> np.ndarray:
    # Initialize the Monte Carlo simulation with the specified parameters
    mc = MonteCarloStockSimulator(mu, sigma, time_horizon)
    
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

