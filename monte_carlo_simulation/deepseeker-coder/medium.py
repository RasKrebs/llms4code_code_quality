import numpy as np
from typing import Tuple

class MonteCarlo:
    """
    A class to simulate stock price paths using the geometric Brownian motion model.

    Attributes:
        annual_return (float): The annual return of the stock.
        volatility (float): The volatility of the stock.
        time (float): The time horizon for the simulation.
    """

    def __init__(self, annual_return: float, volatility: float, time: float):
        """
        The constructor for the MonteCarlo class.

        Parameters:
            annual_return (float): The annual return of the stock.
            volatility (float): The volatility of the stock.
            time (float): The time horizon for the simulation.
        """
        self.annual_return = annual_return
        self.volatility = volatility
        self.time = time

    def simulate(self, initial_price: float, simulation_count: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate stock price paths using the geometric Brownian motion model.

        Parameters:
            initial_price (float): The initial price of the stock.
            simulation_count (int): The number of simulations to run.

        Returns:
            A tuple containing the simulated prices and the time points.
        """
        dt = self.time / simulation_count  # length of time period
        price_paths = np.zeros((simulation_count + 1, simulation_count))
        time_points = np.linspace(0, self.time, simulation_count + 1)

        # Set initial price
        price_paths[0, :] = initial_price

        # Generate random numbers
        random_numbers = np.random.standard_normal((simulation_count + 1, simulation_count))

        # Generate stock prices
        for t in range(1, simulation_count + 1):
            price_paths[t, :] = price_paths[t - 1, :] * np.exp(
                (self.annual_return - 0.5 * self.volatility ** 2) * dt + self.volatility * np.sqrt(dt) * random_numbers[t, :])

        return price_paths, time_points
