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



from scipy import stats
import numpy as np

def monte_carlo(n=10000, initial_price=50, daily_return=0.02, volatility=0.4, time_horizon=365):
    """Run Monte Carlo simulation to estimate stock prices at different points in the future.
    Args:
        n (int): Number of Monte Carlo samples. Defaults to 10000.
        initial_price (float): Initial price of the stock. Defaults to 50.
        daily_return (float): Expected daily return as a decimal percentage. Defaults to 0.02 (2%).
        volatility (float): Volatility expressed as standard deviation. Defaults to 0.4.
        time_horizon (int): Time horizon in days for the simulation. Defaults to 365.
    Returns:
        prices (numpy array): Array of simulated stock prices, with shape (time_horizon+1, n).
            - Row i represents the simulated prices at day i of the simulation.
            - Column j represents the price for the jth sample path.
    """
    # Generate daily returns using log-normal distribution
    daily_returns = stats.lognorm(scale=np.exp(daily_return), s=volatility).rvs((time_horizon, n))

    # Convert daily returns to price paths by taking cumulative product (assume no dividends)
    prices = initial_price * np.cumprod(daily_returns, axis=0)

    return prices


# Execute function
def execute(mu: float, sigma: float, time_horizon: int, time_steps: int,
            initial_stock_price: float, num_simulations: int, random_seed: int) -> np.ndarray:
    # Generate daily returns using log-normal distribution
    daily_returns = stats.lognorm(scale=np.exp(mu), s=sigma).rvs((time_steps, num_simulations))
    # Convert daily returns to price paths by taking cumulative product (assume no dividends)
    prices = initial_stock_price * np.cumprod(daily_returns, axis=0)
    return prices



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

