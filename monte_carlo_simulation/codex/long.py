import numpy as np
from typing import List

class MonteCarlo:
    def __init__(self, expected_return: float, volatility: float, time_horizon: float, time_steps: int):
        """
        Initialize the MonteCarlo class with the given parameters.

        mes:param expected_return: The expected return of the stock.
        mes:param volatility: The volatility of the stock.
        :param time_horizon: The time horizon for the simulation.
        :param time_steps: The number of time steps for the simulation.
        """
        if time_steps <= 0:
            raise ValueError("time_steps must be greater than 0")
        if time_horizon <= 0:
            raise ValueError("time_horizon must be greater than 0")
        if volatility < 0:
            raise ValueError("volatility must be non-negative")

        self.expected_return = expected_return
        self.volatility = volatility
        self.time_horizon = time_horizon
        self.time_steps = time_steps
        self.time_step_size = self.time_horizon / self.time_steps

    def simulate(self, initial_stock_price: float, num_simulations: int, random_seed: int = None) -> np.ndarray:
        """
        Simulate the stock price paths.

        :param initial_stock_price: The initial stock price.
        :param num_simulations: The number of simulations to run.
        :param random_seed: The random seed for reproducibility.
        :return: A numpy array of stock price paths.
        """
        if initial_stock_price <= 0:
            raise ValueError("initial_stock_price must be greater than 0")
        if num_simulations <= 0:
            raise ValueError("num_simulations must be greater than 0")

        if random_seed is not None:
            np.random.seed(random_seed)

        stock_paths = np.zeros((self.time_steps + 1, num_simulations))
        stock_paths[0] = initial_stock_price

        for t in range(1, self.time_steps + 1):
            brownian = np.random.standard_normal(num_simulations)
            stock_paths[t] = (stock_paths[t - 1] * np.exp((self.expected_return - 0.5 * self.volatility ** 2) * self.time_step_size
                                                          + self.volatility * np.sqrt(self.time_step_size) * brownian))

        return stock_paths
