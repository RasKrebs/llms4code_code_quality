import numpy as np
from typing import Tuple

class MonteCarlo:
    """A class for simulating stock price paths using Monte Carlo simulation."""

    def __init__(self, mu: float, sigma: float, T: int):
        self.mu = mu
        self.sigma = sigma
        self.T = T

def simulate(self, initial_price: float, simulation_count: int) -> Tuple[np.ndarray, np.ndarray]:
    """Run Monte Carlo simulations to generate stock price paths.

    Args:
        initial_price (float): Initial price of the underlying asset.
        simulation_count (int): Number of simulations to run.

    Returns:
        time_steps (np.ndarray): Array of time steps for each simulation.
        paths (np.ndarray): Simulated price paths as a 2-D array, with rows representing individual paths and columns representing time steps.
    """

    # Set up the parameters
    dt = 1 / self.T
    count = np.arange(simulation_count)
    time_steps = dt * np.arange(self.T + 1)

    # Generate standard normal random numbers for simulating price paths
    Z = np.random.standard_normal((self.T + 1, simulation_count))

    # Simulate the price paths
    paths = initial_price * np.exp((self.mu - self.sigma**2 / 2) * time_steps + self.sigma * np.sqrt(dt) * Z)