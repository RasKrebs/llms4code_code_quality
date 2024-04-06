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

def simulate_stock_price(S0, mu, sigma, T, dt):
    N = round(T/dt) # number of time steps
    t = np.linspace(0, T, N+1)  # discretized time array from 0 to T with step size dt
    S = np.zeros(N+1)  # stock price process

    # initial condition
    S[0] = S0

    for i in range(1, int(N + 1)):
        eps = np.random.normal()  # create random number from standard normal distribution
        S[i] = S[i - 1] * np.exp((mu - (sigma ** 2) / 2) * dt + sigma * np.sqrt(dt) * eps)

    return t, S

# Execute function
def execute(mu: float, sigma: float, time_horizon: int, time_steps: int,
            initial_stock_price: float, num_simulations: int, random_seed: int) -> np.ndarray:
    # Perform the simulation
    t, S = simulate_stock_price(initial_stock_price, mu, sigma, time_horizon, time_horizon/time_steps)
    
    # return simulated_stock_paths



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

