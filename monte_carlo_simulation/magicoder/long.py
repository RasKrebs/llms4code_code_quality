import numpy as np
from typing import Optional

class StockMonteCarlo:
    """This is a Monte Carlo simulation class to simulate stock prices over time."""

    def __init__(self, expected_return: float, volatility: float,
                 time_horizon: int, time_steps: int) -> Optional[int]:
        self.expected_return = expected_return
        self.volatility = volatility
        if time_horizon <= 0 or time_steps <= 0:
            raise ValueError("Time horizon and steps must be positive integers.")

        self.time_horizon = float(time_horizon)
        self.time_steps = int(time_steps)
        self.delta_t = self.time_horizon / self.time_steps  # type: ignore

    def simulate(self, initial_stock_price: float, num_simulations: int, seed: Optional[int]=None) -> np.ndarray:
        """Generates the stock price paths using Monte Carlo simulation"""

        if initial_stock_price <= 0 or num_simulations <= 0:
            raise ValueError("Initial stock price and number of simulations must be positive integers.")

        # Generate random numbers with normal distribution.
        np.random.seed(seed)  # for reproducibility, set the seed

        dt = self.delta_t
        drift = (self.expected_return - 0.5 * self.volatility**2) * dt
        vol = self.volatility * np.sqrt(dt) * np.random.randn(num_simulations, self.time_steps)

        paths = np.empty((num_simulations, self.time_steps + 1))
        paths[:, 0] = initial_stock_price

        # Calculate each step of the path using Brownian motion
        for t in range(self.time_steps):
            paths[:, t+1] = paths[:, t] * np.exp(drift + vol[:, t])

        return paths
