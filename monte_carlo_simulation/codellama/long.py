import numpy as np
from scipy.stats import norm

class MonteCarlo:
    def __init__(self, expected_return: float, volatility: float, time_horizon: int, time_steps: int):
        """
        Initializes the Monte Carlo simulation parameters.

        Args:
            expected_return (float): Expected return of the asset per unit time
            volatility (float): Volatility of the asset's returns
            time_horizon (int): Time horizon for the simulation in years
            time_steps (int): Number of steps to discretize the simulation into
        """
        if expected_return < 0 or volatility < 0:
            raise ValueError("Expected return and volatility must be positive.")

        self.expected_return = expected_return
        self.volatility = volatility
        self.time_horizon = time_horizon
        self.time_steps = time_steps
        self.time_step_size = time_horizon / time_steps  # Size of each time step in years

    def simulate(self, initial_price: float, num_simulations: int, seed: int) -> np.ndarray:
        """
        Simulates stock price paths using a Brownian motion process (also known as a Wiener process).

        Args:
            initial_price (float): Initial price of the asset at time 0
            num_simulations (int): Number of simulations to generate
            seed (int): Random seed for reproducibility

        Returns:
            np.ndarray: A numpy array of shape (num_simulations, time_steps + 1) containing the simulated stock price paths.
        """

        # Initialize the simulation matrix with zeros
        simulations = np.zeros((num_simulations, self.time_steps + 1))

        # Set the initial prices
        simulations[:, 0] = initial_price

        # Generate random standard normal variates for the Brownian motion process
        np.random.seed(seed)
        brownian_motion = norm.rvs(size=(num_simulations, self.time_steps))

        for i in range(1, self.time_steps + 1):
            # Simulate the stock price at time step i using a Brownian motion process:
            # S[i] = S[i-1] * exp((r - 0.5*sigma^2)*dt + sigma * sqrt(dt) * Z[i])

            simulations[:, i] = (
                simulations[:, i-1] * np.exp(
                    (self.expected_return - 0.5 * self.volatility ** 2) * self.time_step_size +
                    self.volatility * np.sqrt(self.time_step_size) * brownian_motion[:, i-1]
                )
            )

        return simulations