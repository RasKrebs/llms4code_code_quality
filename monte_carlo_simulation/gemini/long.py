import numpy as np


class MonteCarlo:
    """
    This class simulates stock price paths using a Geometric Brownian Motion model.

    Attributes:
        expected_return (float): The expected return of the stock.
        volatility (float): The volatility of the stock.
        time_horizon (float): The time horizon of the simulation in years.
        time_steps (int): The number of time steps for the simulation.
        time_step_size (float): The size of each time step (years).
    """

    def __init__(self, expected_return: float, volatility: float, time_horizon: float, time_steps: int) -> None:
        """
        Initializes a MonteCarlo object.

        Args:
            expected_return (float): The expected return of the stock.
            volatility (float): The volatility of the stock.
            time_horizon (float): The time horizon of the simulation in years.
            time_steps (int): The number of time steps for the simulation.

        Raises:
            ValueError: If expected return, volatility, time horizon, or time steps are non-positive.
        """

        if expected_return <= 0:
            raise ValueError("Expected return must be positive.")
        if volatility <= 0:
            raise ValueError("Volatility must be positive.")
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
        Simulates stock price paths using Geometric Brownian Motion.

        Args:
            initial_price (float): The initial price of the stock.
            num_simulations (int): The number of simulations to run.
            random_seed (int, optional): The random seed for reproducibility. Defaults to None.

        Returns:
            numpy.ndarray: A numpy array of shape (num_simulations, time_steps + 1) containing the 
                           simulated stock price paths. The first column represents the initial price.
        """

        if initial_price <= 0:
            raise ValueError("Initial price must be positive.")
        if num_simulations <= 0:
            raise ValueError("Number of simulations must be positive.")

        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)

        # Drift and volatility terms for geometric Brownian motion
        drift = self.expected_return - (self.volatility**2) / 2
        volatility_term = self.volatility * np.sqrt(self.time_step_size)

        # Generate random shocks for each path
        random_shocks = np.random.normal(scale=volatility_term, size=(num_simulations, self.time_steps))

        # Initialize price paths with initial price
        price_paths = np.ones((num_simulations, self.time_steps + 1)) * initial_price

        # Simulate price paths using geometric Brownian motion
        for i in range(1, self.time_steps + 1):
            price_paths[:, i] = price_paths[:, i - 1] * np.exp(drift * self.time_step_size + random_shocks[:, i - 1])

        return price_paths
