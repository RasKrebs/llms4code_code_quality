import numpy as np


class MonteCarlo:
    """
    This class simulates stock price paths using a Geometric Brownian Motion (GBM) model.

    Attributes:
        annual_return (float): The expected annual return of the stock.
        volatility (float): The annual volatility of the stock.
        time_horizon (float): The time horizon of the simulation in years.
    """

    def __init__(self, annual_return: float, volatility: float, time_horizon: float) -> None:
        """
        Initializes the MonteCarlo class with the given parameters.

        Args:
            annual_return (float): The expected annual return of the stock.
            volatility (float): The annual volatility of the stock.
            time_horizon (float): The time horizon of the simulation in years.
        """
        self.annual_return = annual_return
        self.volatility = volatility
        self.time_horizon = time_horizon

    def simulate(self, initial_price: float, num_simulations: int) -> np.ndarray:
        """
        Simulates stock price paths using the GBM model.

        Args:
            initial_price (float): The initial price of the stock.
            num_simulations (int): The number of price paths to simulate.

        Returns:
            np.ndarray: A numpy array of shape (num_simulations, time_steps) containing the simulated price paths.
        """

        # Calculate drift and diffusion terms for the GBM model
        dt = self.time_horizon / 252  # Time step (assuming 252 trading days per year)
        drift = (self.annual_return - 0.5 * self.volatility**2) * dt
        diffusion = self.volatility * np.sqrt(dt)

        # Simulate random Wiener process increments
        random_increments = np.random.normal(scale=diffusion, size=(num_simulations, int(self.time_horizon * 252)))

        # Initialize price paths
        price_paths = np.empty((num_simulations, int(self.time_horizon * 252) + 1))
        price_paths[:, 0] = initial_price

        # Simulate price paths using GBM formula
        for i in range(1, int(self.time_horizon * 252) + 1):
            price_paths[:, i] = price_paths[:, i - 1] * np.exp(drift + random_increments[:, i - 1])

        return price_paths
