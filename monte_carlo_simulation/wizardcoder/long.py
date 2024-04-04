import numpy as np
from typing import Optional

class MonteCarlo:
    def __init__(self, expected_return: float, volatility: float, time_horizon: float, time_steps: int):
        """
        Initialize a Monte Carlo object with the specified parameters.

        :param expected_return: The average return of the stock (annualized)
        :param volatility: The standard deviation of daily returns (annualized)
        :param time_horizon: Total length of simulation, in years
        :param time_steps: Number of steps to use for the simulation

        """
        if expected_return < 0 or volatility < 0 or time_horizon <= 0 or time_steps <= 0:
            raise ValueError("Parameters must be positive.")

        self.expected_return = expected_return
        self.volatility = volatility
        self.time_horizon = time_horizon
        self.time_steps = time_steps
        self.time_step_size = time_horizon / float(time_steps)  # Calculate time step size

    def simulate(self, initial_price: float, num_simulations: int, seed: Optional[int] = None):
        """
        Run a Monte Carlo simulation to generate stock price paths.

        :param initial_price: The starting price of the stock at time 0
        :param num_simulations: Number of simulations to run
        :param seed: An optional random number generator seed for reproducibility
        :return: A NumPy array of simulated stock prices, with shape (num_steps + 1, num_simulations)

        """
        if initial_price <= 0 or num_simulations <= 0:
            raise ValueError("Parameters must be positive.")

        try:
            # Set the seed for the random number generator
            np.random.seed(seed)

            # Define the mean and standard deviation of daily returns
            mu = self.expected_return / self.time_steps
            sigma = self.volatility / np.sqrt(self.time_steps)

            # Generate random daily returns for each time step and simulation
            rand_returns = np.random.normal(loc=mu, scale=sigma, size=(self.time_steps, num_simulations))

            # Calculate the cumulative product to get the paths
            price_paths = initial_price * np.cumprod(1 + rand_returns, axis=0)

            return price_paths
        except Exception as e:
            print("An error occurred during simulation.")
            raise e
