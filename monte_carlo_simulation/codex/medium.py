import numpy as np

class MonteCarlo:
    """
    A class used to simulate stock price paths using the Monte Carlo method.

    Attributes
    ----------
    annual_return : float
        The expected annual return of the stock.
    volatility : float
        The volatility of the stock.
    time : float
        The time horizon for the simulation.

    Methods
    -------
    simulate(initial_price: float, num_simulations: int) -> np.ndarray:
        Simulates the stock price paths.
    """

    def __init__(self, annual_return: float, volatility: float, time: float):
        """
        Parameters
        ----------
        annual_return : float
            The expected annual return of the stock.
        volatility : float
            The volatility of the stock.
        time : float
            The time horizon for the simulation.
        """
        self.annual_return = annual_return
        self.volatility = volatility
        self.time = time

    def simulate(self, initial_price: float, num_simulations: int) -> np.ndarray:
        """
        Simulates the stock price paths.

        Parameters
        ----------
        initial_price : float
            The initial price of the stock.
        num_simulations : int
            The number of simulations to run.

        Returns
        -------
        np.ndarray
            An array of simulated stock price paths.
        """
        # Calculate the daily return
        daily_return = self.annual_return / 252

        # Initialize the price paths array
        price_paths = np.zeros((252 * int(self.time), num_simulations))

        # Set the initial prices
        price_paths[0] = initial_price

        # Simulate the price paths
        for t in range(1, 252 * int(self.time)):
            brownian = np.random.normal(0, 1, num_simulations)
            price_paths[t] = price_paths[t - 1] * np.exp((daily_return - 0.5 * self.volatility**2) +
                                                         self.volatility * brownian)

        return price_paths
