import numpy as np
from typing import List, Tuple

class MonteCarlo:
    """
    This class is created to simulate stock price paths based on the Black-Scholes model.
    It includes attributes for annual return, volatility, and time parameters
    along with a simulate function to generate price paths.
    """

    def __init__(self, annual_return: float, volatility: float, time: int):
        self.annual_return = annual_return
        self.volatility = volatility
        self.time = time

    def simulate(self, initial_price: float, simulation_count: int) -> List[List[float]]:
        """
        Simulates the stock price paths based on the Black-Scholes model.

        Parameters
        ----------
        initial_price : float
            The starting stock price.
        simulation_count : int
            The number of simulations to run.

        Returns
        -------
        List[List[float]]
            A list containing the simulated stock price paths for each time step.

        Examples
        --------
        >>> mc = MonteCarlo(0.05, 0.2, 1)
        >>> paths = mc.simulate(100, 1000)
        >>> print(paths[0])  # The first path will have the initial price followed by simulated prices for each time step.
        [100, ..., ...]
        """
        dt = 1/365  # Daily return
        paths = []  # List to store all the simulations

        # Simulating each path
        for _ in range(simulation_count):
            price_path = [initial_price]  # Starting with initial stock price

            # Generating each time step
            for t in range(1, self.time + 1):
                drift = (self.annual_return - 0.5 * self.volatility**2) * dt
                shock = np.random.normal(loc=0, scale=self.volatility*np.sqrt(dt))

                price = price_path[-1] * np.exp(drift + shock)
                price_path.append(price)  # Appending the new stock price to the path

            paths.append(price_path)

        return paths
