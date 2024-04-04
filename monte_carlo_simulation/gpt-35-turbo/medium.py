class MonteCarloSimulation:
    def __init__(self, annual_return: float, volatility: float, time_period: int):
        """
        Initialize the Monte Carlo Simulation with parameters for annual return,
        volatility, and time period.
        
        Parameters:
        annual_return (float): The average annual return of the stock
        volatility (float): The volatility of the stock
        time_period (int): The number of days to simulate
        """
        self.annual_return = annual_return
        self.volatility = volatility
        self.time_period = time_period
    
    def simulate(self, initial_price: float, num_simulations: int) -> np.array:
        """
        Simulate stock price paths using a Monte Carlo simulation.
        
        Parameters:
        initial_price (float): The initial price of the stock
        num_simulations (int): The number of simulations to run
        
        Returns:
        paths (np.array): An array of stock price paths (num_simulations x time_period)
        """
        paths = np.zeros((num_simulations, self.time_period))
        time = np.arange(1, self.time_period + 1)
        
        for i in range(num_simulations):
            path = np.zeros(self.time_period)
            path[0] = initial_price
            
            for t in range(1, self.time_period):
                daily_return = np.random.normal(self.annual_return/self.time_period, self.volatility)
                path[t] = path[t-1] * (1 + daily_return)
            
            paths[i, :] = path
        
        return paths
