import numpy as np

class MonteCarloStockSimulator:
    """
    A class to simulate stock price paths using the Monte Carlo method.
    
    Attributes:
        annual_return (float): The expected annual return of the stock.
        volatility (float): The annual volatility of the stock's returns.
        time_horizon (int): The time horizon for the simulation in years.
        
    Methods:
        simulate(initial_price: float, n_simulations: int) -> np.ndarray:
            Generates simulated stock price paths.
    """
    
    def __init__(self, annual_return: float, volatility: float, time_horizon: int):
        """
        Initializes the MonteCarloStockSimulator with expected annual return, 
        volatility, and time horizon for the simulation.
        
        Parameters:
            annual_return (float): The expected annual return of the stock.
            volatility (float): The annual volatility of the stock returns.
            time_horizon (int): The time horizon for the simulation in years.
        """
        self.annual_return = annual_return
        self.volatility = volatility
        self.time_horizon = time_horizon
        
    def simulate(self, initial_price: float, n_simulations: int) -> np.ndarray:
        """
        Simulates stock price paths over the specified time horizon.
        
        Using the Geometric Brownian Motion model for stock price evolution,
        this function generates several simulated paths for the stock price, 
        based on its annual return, volatility, and a given initial price.
        
        Parameters:
            initial_price (float): The starting price of the stock.
            n_simulations (int): The number of simulated paths to generate.
        
        Returns:
            np.ndarray: A 2D array where each row represents a simulated stock price path.
        """
        # Daily returns are used for simulation
        days = self.time_horizon * 252  # assuming 252 trading days in a year
        daily_return = self.annual_return / 252
        daily_volatility = self.volatility / np.sqrt(252)
        
        # Random component in Geometric Brownian Motion
        random_shocks = np.random.normal(loc=daily_return, scale=daily_volatility, size=(days, n_simulations))
        
        # Calculating price paths
        price_paths = np.zeros_like(random_shocks)
        price_paths[0] = initial_price
        
        for t in range(1, days):
            price_paths[t] = price_paths[t-1] * np.exp(random_shocks[t])
        
        return price_paths
