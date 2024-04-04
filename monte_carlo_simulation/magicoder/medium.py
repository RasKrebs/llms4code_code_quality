import numpy as np
from typing import Optional

class MonteCarloSimulation:
    """
    A class to simulate stock price paths with parameters for initial price,
    annual return, volatility and time.
    """

    def __init__(self,
                 mean_return: float,
                 volatility: float,
                 time: Optional[float] = None):
        """
        Initialize MonteCarloSimulation object with parameters for mean return,
        volatility and time. If no time is provided it defaults to 1 year.

        Parameters:
            mean_return (float): annual return rate of the stock
            volatility (float): standard deviation of price changes
            time (Optional[float] = None) -> Optional[float]: if not provided, default time is one year

        time: float = 1.0
        """
        self.mean_return = mean_return
        self.volatility = volatility
        self.time = time if time else 1.0

    def simulate(self, initial_price: float, simulations: int) -> np.ndarray:
        """
        Simulate stock price paths for a specified number of simulations and return the result as an array.

        Parameters:
            initial_price (float): initial price of the stock
            simulations (int): number of price path simulations to run

        Returns:
            np.ndarray: numpy array containing all simulated prices for each simulation
        """
        dt = 1 / self.time
        random_numbers = np.random.normal(0, dt ** 0.5,(simulations, int(self.time)))

        # Calculate price paths using the formula pt = pte * e^((r - sigma^2/2)t + sigma * dWt)
        prices = initial_price * np.exp(np.cumsum((self.mean_return - 0.5 * self.volatility ** 2) * dt \
                                                  + self.volatility * random_numbers, axis=1))

        return prices
