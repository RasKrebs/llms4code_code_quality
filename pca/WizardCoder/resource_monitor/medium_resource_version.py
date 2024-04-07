import psutil
import os
import threading
import multiprocessing

# Define a global variable to store the maximum resources usage
max_resources_usage = {"cpu": 0, "memory": 0}


import numpy as np
np.random.seed(42)


# Implement the resource monitor
def resource_monitor():
    """
    Monitors the CPU and memory usage of the current process, updating global max usage.
    """
    global max_resources_usage
    process = psutil.Process(os.getpid())

    while monitoring:
        cpu_usage = process.cpu_percent(interval=1) / multiprocessing.cpu_count()
        memory_usage = process.memory_info().rss
        max_resources_usage['cpu'] = max(max_resources_usage['cpu'], cpu_usage)
        max_resources_usage['memory'] = max(max_resources_usage['memory'], memory_usage)



import numpy as np
from sklearn import datasets
from typing import Union

class PrincipalComponentAnalysis:
    def __init__(self, n_components: int = None) -> None:
        """Initialize the PCA object with number of components to be retained.

        Args:
            n_components (int): Number of components to retain. Defaults to all components.

        Returns:
            None
        """
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit(self, X: np.ndarray) -> 'PrincipalComponentAnalysis':
        """Fit the PCA model with X by calculating mean and components.

        Args:
            X (np.ndarray): Data to fit the model.

        Returns:
            self (PrincipalComponentAnalysis): The fitted PrincipalComponentAnalysis object.
        """

        # Calculate the mean of each feature
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # Calculating the covariance matrix of the mean-centered data.
        cov_mat = np.cov(X.T)

        # Compute eigen values and eigen vectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_mat)

        # Sort eigenvalues in descending order and sort the corresponding eigenvectors
        sorted_index = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_index]
        sorted_eigenvectors = eigenvectors[:,sorted_index]

        # Selecting the first n eigenvectors as per given dimension
        if self.n_components is not None:
            sorted_eigenvectors = sorted_eigenvectors[:, :self.n_components]

        # Store the resultant vectors
        self.components = sorted_eigenvectors

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the given data using the fitted PCA model.

        Args:
            X (np.ndarray): Data to be transformed.

        Returns:
            X_transformed (np.ndarray): Transformed data.
        """

        # Mean centering
        X = X - self.mean

        # Return the projection of X onto the principal components
        return np.dot(X, self.components)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform the data using the fitted PCA model.

        Args:
            X (np.ndarray): Data to be transformed.

        Returns:
            X_transformed (np.ndarray): Transformed data.
        """
        return self.fit(X).transform(X)

def execute():
    # Set the random seed for reproducibility
    np.random.seed(42)

    # Generate random data: 10000 samples with 1000 features
    X = np.random.rand(10000, 1000)

    # Initialize PCA with 2 components
    pca = PrincipalComponentAnalysis(n_components=2)

    # Fit PCA on the generated data
    pca.fit(X)

    # Transform the data using the fitted PCA
    X_pca = pca.transform(X)

# Execute the function to see the results



if __name__ == "__main__":
    # Start the resource monitoring in a separate thread
    global monitoring
    monitoring = True
    monitor_thread = threading.Thread(target=resource_monitor)
    monitor_thread.start()

    # Execute the Huffman coding process

    # Using the execute function
    output = execute()


    # Stop the monitoring
    monitoring = False
    monitor_thread.join()

    print(max_resources_usage['cpu']), print(max_resources_usage['memory'] / (1024**2))

