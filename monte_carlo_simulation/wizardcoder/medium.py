import numpy as np
from typing import List

class MonteCarloStockPrice:
    def __init__(self, annual_return: float, volatility: float, time: int) -> None:
        """
        Initialize the class with parameters.

        Parameters:
            annual_return (float): The expected return of the stock annually.
            volatility (float): The standard deviation of log returns.
            time (int): The total number of periods for which to simulate prices.
        """
        self.annual_return = annual_return
        self.volatility = volatility
        self.time = time

    def simulate(self, initial_price: float, num_simulations: int) -> List[List[float]]:
        """
        Generate a number of price paths based on the Monte Carlo simulation methodology.

        Parameters:
            initial_price (float): The starting price for all simulations.
            num_simulations (int): The total number of simulations to generate.

        Returns:
            A list of lists, where each inner list represents a single simulated path of prices.
        """
        # Define the mean and standard deviation for the normal distribution used in the Monte Carlo simulation
        mu = self.annual_return - (self.volatility ** 2) / 2
        sigma = self.volatility

        dt = 1.0/365  # daily returns
        total_time = 1.0 * self.time

        paths = []
        for _ in range(num_simulations):
            path = [initial_price]

            for _ in np.arange(dt, total_time + dt, dt):  # generate prices for each time step
                rtn = np.random.normal(mu * dt, sigma * np.sqrt(dt))  # daily returns
                price = path[-1] * np.exp(rtn)  # new price
                path.append(price)

            paths.append(path)

        return paths
