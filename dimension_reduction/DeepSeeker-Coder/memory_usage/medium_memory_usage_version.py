from memory_profiler import profile
import numpy as np
from scipy.linalg import svd
from typing import Optional, Union
import psutil
import os

# Get the current process ID
pid = os.getpid()

# Create a psutil Process object for the current process
process = psutil.Process(pid)

# Get the number of logical CPUs in the system
num_cores = psutil.cpu_count(logical=True)

class PrincipalComponentAnalysis:
    """
    A class to perform Principal Component Analysis.
    """
    @profile
    def __init__(self, n_components: Optional[int] = None, decomposition_method: Optional[str] = 'eigen'):
        """
        Initialize the PrincipalComponentAnalysis object.

        Parameters:
        n_components (int, optional): Number of components to use. Defaults to all.
        decomposition_method (str, optional): Decomposition method to use. Can be 'eigen' or 'svd'. Defaults to 'eigen'.
        """
        self.n_components = n_components
        self.decomposition_method = decomposition_method
        self.mean = None
        self.components = None
    @profile
    def fit(self, X: np.ndarray) -> None:
        """
        Fit the PCA model with X.

        Parameters:
        X (np.ndarray): The data to fit the model with.
        """
        # Center the data
        self.mean = np.mean(X, axis=0)
        X -= self.mean

        # Compute the covariance matrix
        C = np.cov(X.T)

        # Compute the eigenvalues and eigenvectors
        if self.decomposition_method == 'eigen':
            eigenvalues, eigenvectors = np.linalg.eig(C)
        elif self.decomposition_method == 'svd':
            _, eigenvalues, eigenvectors = svd(C)
        else:
            raise ValueError("Invalid decomposition method. Choose from 'eigen' or 'svd'.")

        # Sort the eigenvectors by decreasing eigenvalues
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Select the first n_components eigenvectors
        if self.n_components is not None:
            eigenvectors = eigenvectors[:, :self.n_components]

        self.components = eigenvectors

    @profile
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply dimensionality reduction to X.

        Parameters:
        X (np.ndarray): The data to transform.

        Returns:
        np.ndarray: The transformed data.
        """
        X -= self.mean
        return np.dot(X, self.components)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the PCA model with X and apply the dimensionality reduction on X.

        Parameters:
        X (np.ndarray): The data to fit the model with and transform.

        Returns:
        np.ndarray: The transformed data.
        """
        self.fit(X)
        return self.transform(X)

# Execute script data to get memory usage
if __name__ == "__main__":
    # For reproducibility
    np.random.seed(42)

    # Generate random data
    data = np.random.rand(10000, 100)
    
    # Create PCA object
    pca = PrincipalComponentAnalysis(n_components=5)

    # Fit and transform data
    _ = pca.fit_transform(data)
    
    # Get the CPU percentage usage of the process
    cpu_usage = process.cpu_percent(interval=1)/ num_cores
    print(f"CPU Usage: {cpu_usage}%")
