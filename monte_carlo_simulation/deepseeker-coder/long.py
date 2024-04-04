import numpy as np
from typing import Tuple

class MonteCarlo:
    """
    A class to simulate stock price paths using the Monte Carlo method.

    Attributes:
        expected_return (float): The expected return of the stock.
        volatility (float): The volatility of the stock.
        time_horizon (float): The time horizon for the simulation.
        time_steps (int): The number of time steps for the simulation.
        time_step_size (float): The size of each time step.
    """

    def __init__(self, expected_return: float, volatility: float, time_horizon: float, time_steps: int):
        """
        The constructor for the MonteCarlo class.

        Args:
            expected_return (float): The expected return of the stock.
            volatility (float): The volatility of the stock.
            time_horizon (float): The time horizon for the simulation.
            time_steps (int): The number of time steps for the simulation.
        """

        # Validate inputs
        if expected_return < 0:
            raise ValueError("Expected return must be non-negative.")
        if volatility < 0:
            raise ValueError("Volatility must be non-negative.")
        if time_horizon <= 0:
            raise ValueError("Time horizon must be positive.")
        if time_steps <= 0:
            raise ValueError("Time steps must be positive.")

        self.expected_return = expected_return
        self.volatility = volatility
        self.time_horizon = time_horizon
        self.time_steps = time_steps
        self.time_step_size = time_horizon / time_steps

    def simulate(self, initial_price: float, num_simulations: int, random_seed: int = None) -> np.ndarray:
        """
        Simulate stock price paths using the Monte Carlo method.

        Args:
            initial_price (float): The initial price of the stock.
            num_simulations (int): The number of simulations to run.
            random_seed (int, optional): The random seed for reproducibility. Defaults to None.

        Returns:
            np.ndarray: A numpy array of stock price paths.
        """

        # Validate inputs
        if initial_price <= 0:
            raise ValueError("Initial price must be positive.")
        if num_simulations <= 0:
            raise ValueError("Number of simulations must be positive.")

        # Set the random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)

        # Initialize the price paths array
        price_paths = np.zeros((num_simulations, self.time_steps + 1))
        price_paths[:, 0] = initial_price

        # Generate the price paths
        for t in range(1, self.time_steps + 1):
            # Generate random returns
            returns = np.random.normal(self.expected_return, self.volatility, num_simulations)

            # Update the price paths
            price_paths[:, t] = price_paths[:, t - 1] * np.exp(returns)

        return price_paths

