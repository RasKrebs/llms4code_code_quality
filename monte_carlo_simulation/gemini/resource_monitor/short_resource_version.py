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

def simulate_stock_prices(init_price, expected_return, volatility, time_horizon, steps):
  """
  Simulates stock prices using Monte Carlo simulation.

  Args:
      init_price: Initial price of the stock.
      expected_return: Expected daily return of the stock.
      volatility: Volatility of the stock (annualized standard deviation).
      time_horizon: Time horizon for the simulation (in days).
      steps: Number of steps in the simulation (including the initial price).

  Returns:
      A numpy array of simulated stock prices for each time step.
  """
  # Daily volatility (assuming constant volatility)
  daily_volatility = volatility / np.sqrt(252)  # 252 trading days per year

  # Simulate daily log returns using a normal distribution
  daily_log_returns = np.random.normal(loc=expected_return, scale=daily_volatility, size=steps)

  # Calculate daily price changes from log returns
  daily_price_changes = np.exp(daily_log_returns) - 1

  # Initialize price path
  price_paths = np.zeros(steps)
  price_paths[0] = init_price

  # Simulate stock price for each time step
  for i in range(1, steps):
    price_paths[i] = price_paths[i-1] * (1 + daily_price_changes[i])

  return price_paths

# Execute function
def execute(mu: float, sigma: float, time_horizon: int, time_steps: int,
      initial_stock_price: float, num_simulations: int, random_seed: int) -> np.ndarray:
  """
  Executes the Monte Carlo simulation for stock prices.

  Args:
    mu: Expected annual return.
    sigma: Volatility.
    time_horizon: Time horizon in years.
    time_steps: Number of time steps, assuming 252 trading days in a year.
    initial_stock_price: Initial stock price.
    num_simulations: Number of simulations.
    random_seed: Random seed for reproducibility.

  Returns:
    A numpy array of simulated stock prices for each time step.
  """
  # Load data/parameters section
  np.random.seed(random_seed)
  
  # Daily volatility (assuming constant volatility)
  daily_volatility = sigma / np.sqrt(252)  # 252 trading days per year
  
  # Simulate stock prices using Monte Carlo simulation
  simulated_stock_paths = simulate_stock_prices(initial_stock_price, mu, daily_volatility, time_horizon, time_steps)
  
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

