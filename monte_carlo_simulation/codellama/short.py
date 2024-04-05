from scipy import stats
import numpy as np

def monte_carlo(n=10000, initial_price=50, daily_return=0.02, volatility=0.4, time_horizon=365):
    """Run Monte Carlo simulation to estimate stock prices at different points in the future.
    Args:
        n (int): Number of Monte Carlo samples. Defaults to 10000.
        initial_price (float): Initial price of the stock. Defaults to 50.
        daily_return (float): Expected daily return as a decimal percentage. Defaults to 0.02 (2%).
        volatility (float): Volatility expressed as standard deviation. Defaults to 0.4.
        time_horizon (int): Time horizon in days for the simulation. Defaults to 365.
    Returns:
        prices (numpy array): Array of simulated stock prices, with shape (time_horizon+1, n).
            - Row i represents the simulated prices at day i of the simulation.
            - Column j represents the price for the jth sample path.
    """
    # Generate daily returns using log-normal distribution
    daily_returns = stats.lognorm(scale=np.exp(daily_return), s=volatility).rvs((time_horizon, n))

    # Convert daily returns to price paths by taking cumulative product (assume no dividends)
    prices = initial_price * np.cumprod(daily_returns, axis=0)

    return prices

