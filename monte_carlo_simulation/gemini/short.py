import numpy as np

def simulate_stock_prices(init_price, expected_return, volatility, time_horizon, steps):
    """
    Simulates stock prices using Monte Carlo simulation.

    Args:
        init_price: Initial price of the stock.
        expected_return: Expected daily return of the stock.
        volatility: Volatility of the stock (annualized standard deviation).
        time_horizon: Time horizon for the simulation (in days).
        steps: Number of steps in the simulation (including the initial price).

    Returns:
        A numpy array of simulated stock prices for each time step.
    """
    # Daily volatility (assuming constant volatility)
    daily_volatility = volatility / np.sqrt(252)  # 252 trading days per year

    # Simulate daily log returns using a normal distribution
    daily_log_returns = np.random.normal(loc=expected_return, scale=daily_volatility, size=steps)

    # Calculate daily price changes from log returns
    daily_price_changes = np.exp(daily_log_returns) - 1

    # Initialize price path
    price_paths = np.zeros(steps)
    price_paths[0] = init_price

    # Simulate stock price for each time step
    for i in range(1, steps):
        price_paths[i] = price_paths[i-1] * (1 + daily_price_changes[i])

    return price_paths
