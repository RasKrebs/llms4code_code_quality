"""Monte Carlo simulation module for stock price paths.

This module contains the MonteCarlo class, which is used to simulate stock price paths using Monte 
Carlo simulation.

Example:
    >>> mc = MonteCarlo(0.1, 0.2, 1, 252)
    >>> mc.simulate(100, 1000)
    array([[100.        , 100.        , 100.        , ..., 100.        ,
        ... ]])
            
"""

import numpy as np

class MonteCarlo:
    """
    Class representing a Monte Carlo simulation for stock price paths.

    Attributes:
    - mu (int): Expected annual return.
    - sigma (int): Volatility.
    - time (int): Time horizon in years.
    - time_steps (int): Number of time steps, aka trading days.
    - time_step_size (float): Size of each time step.

    Methods:
    - __init__(expected_return, volatility, time_horizon, time_steps): Initialize the Monte Carlo 
        simulation.
    - simulate(initial_stock_price, num_simulations, random_seed): Simulate stock price paths using 
        Monte Carlo simulation.
    """

    def __init__(self,
                 expected_return:int,
                 volatility:int,
                 time_horizon:int,
                 time_steps:int):
        """
        Initialize Monte Carlo simulation with the following parameters:

        Parameters:
        - expected_return (int): Expected annual return.
        - volatility (int): Volatility.
        - time_horizon (int): Time horizon in years.
        - time_steps (int): Number of time steps, aka trading days.
        """
        self.mu = expected_return       # Expected annual return
        self.sigma = volatility         # Volatility
        self.time = time_horizon        # Time horizon in years
        self.time_steps = time_steps    # Number of time steps, aka trading
                                        # days
        self.time_step_size = self.time/self.time_steps

    def simulate(self,
                 initial_stock_price:int,
                 num_simulations:int,
                 random_seed:int = 42) -> np.ndarray:
        """
        Simulate stock price paths using Monte Carlo simulation.

        Parameters:
        - initial_stock_price (int): Initial stock price.
        - num_simulations (int): Number of simulations.
        - random_seed (int): Random seed for reproducibility.

        Returns:
        - output (np.ndarray): Array of simulated stock price paths.
        """
        # Random seed
        np.random.seed(random_seed)

        # Simulating M paths with N time steps
        dist = np.random.standard_normal((self.time_steps, num_simulations))

        # Output array
        output = np.zeros((self.time_steps+1, num_simulations))

        # Set initial stock price at time 0
        output[0] = initial_stock_price

        # Generate paths
        for time in range(1, self.time_steps+1):
            # Calculate the next stock price based on the previous price and random normal
            # distribution
            output[time] = output[time-1] * np.exp((self.mu - 0.5 * self.sigma**2) *
                                                   self.time_step_size + self.sigma *
                                                   np.sqrt(self.time_step_size) * dist[time-1])

        return output
