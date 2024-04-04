import numpy as np
from typing import Tuple, List

class MonteCarlo:
    def __init__(self, expected_return: float, volatility: float, time_horizon: int, time_steps: int) -> None:
        self.expected_return = expected_return
        self.volatility = volatility
        self.time_horizon = time_horizon
        self.time_steps = time_steps
        self.time_step_size = self.time_horizon / self.time_steps  # Calculate the size of each time step

    def simulate(self, initial_price: float, num_simulations: int, seed: int) -> np.ndarray:
        try:
            # Seed the random number generator for reproducibility
            np.random.seed(seed)

            stock_paths = []  # List to store all simulated paths

            # For each simulation
            for _ in range(num_simulations):
                path = [initial_price]  # Initialize the path with initial price

                # Simulate each time step
                for _ in range(self.time_steps):
                    random_number = np.random.normal()  # Generate a random number from standard normal distribution

                    # Update the next stock price based on the expected return, volatility and random number
                    next_price = path[-1] * (1 + self.expected_return * self.time_step_size + self.volatility * np.sqrt(self.time_step_size) * random_number)

                    # Append the new stock price to the path
                    path.append(next_price)

                # Append the completed path to the list of paths
                stock_paths.append(path)

            return np.array(stock_paths)  # Convert the list of paths into a numpy array and return it

        except Exception as e:
            raise ValueError('An error occurred during simulation: ' + str(e)) from e
