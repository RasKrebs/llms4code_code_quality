import numpy as np
import numpy as np
import numpy.random as npr

class MonteCarlo:
    def __init__(self, expected_return: float, volatility: float, time_horizon: int, time_steps: int):
        """
        Initialize the MonteCarlo simulation with parameters.

        Parameters:
        expected_return (float): The expected return of the stock.
        volatility (float): The volatility of the stock.
        time_horizon (int): The total time duration of the simulation.
        time_steps (int): The number of time steps in the simulation.
        """
        self.expected_return = expected_return
        self.volatility = volatility
        self.time_horizon = time_horizon
        self.time_steps = time_steps
        self.time_step_size = time_horizon / time_steps

        # Validate input parameters
        if self.time_step_size <= 0:
            raise ValueError("Time step size must be greater than 0.")
    
    def simulate(self, initial_stock_price: float, num_simulations: int, random_seed: int = None) -> np.ndarray:
        """
        Generate stock price paths through Monte Carlo simulation.

        Parameters:
        initial_stock_price (float): The initial stock price for simulation.
        num_simulations (int): The number of simulation paths to generate.
        random_seed (int): The random seed for reproducibility.

        Returns:
        np.ndarray: A numpy array of stock price paths with shape (num_simulations, time_steps+1).
        """
        if num_simulations <= 0:
            raise ValueError("Number of simulations must be greater than 0.")
        
        # Set random seed for reproducibility
        if random_seed:
            npr.seed(random_seed)
        
        stock_paths = np.zeros((num_simulations, self.time_steps + 1))
        stock_paths[:, 0] = initial_stock_price

        for t in range(1, self.time_steps + 1):
            # Generate random samples from a normal distribution
            rand_samples = npr.normal(self.expected_return * self.time_step_size, 
                                      self.volatility * np.sqrt(self.time_step_size), 
                                      num_simulations)
            stock_paths[:, t] = stock_paths[:, t - 1] * np.exp(rand_samples)
        
        return stock_paths
