
# IMPORTED FROM EVALUATE SCRIPT
from memory_profiler import profile
import os
import psutil

# Get the current process ID
pid = os.getpid()

# Create a psutil Process object for the current process
process = psutil.Process(pid)

# Get the number of logical CPUs in the system
num_cores = psutil.cpu_count(logical=True)

# BELOW IMPORTS ARE FROM THE SOLUTION
        
import numpy as np

class PrincipalComponentAnalysis:
    """
    Principal Component Analysis (PCA) class.
    PCA is a dimensionality reduction technique that is widely used in practice.
    This class implements PCA using eigenvalue decomposition.
    """

    @profile
    def __init__(self, n_components: int = None):
        """
        Initialize the PCA object.

        :param n_components: The number of components to use. If not specified, all components are used.
        """
        self.n_components = n_components
        self.eigenvalues = None
        self.eigenvectors = None

    @profile
    def fit(self, X: np.ndarray):
        """
        Fit the PCA model with the input data.

        :param X: The input data, a 2D numpy array of shape (n_samples, n_features).
        """
        # Center the data
        X_centered = X - np.mean(X, axis=0)

        # Compute the covariance matrix
        covariance_matrix = np.cov(X_centered.T)

        # Compute the eigenvalues and eigenvectors
        self.eigenvalues, self.eigenvectors = np.linalg.eig(covariance_matrix)

        # Sort the eigenvectors by decreasing eigenvalues
        idx = np.argsort(self.eigenvalues)[::-1]
        self.eigenvalues = self.eigenvalues[idx]
        self.eigenvectors = self.eigenvectors[:, idx]

    @profile
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply dimensionality reduction to the input data.

        :param X: The input data, a 2D numpy array of shape (n_samples, n_features).
        :return: The transformed data.
        """
        # Center the data
        X_centered = X - np.mean(X, axis=0)

        # Project the data onto the principal components
        if self.n_components is None:
            return np.dot(X_centered, self.eigenvectors)
        else:
            return np.dot(X_centered, self.eigenvectors[:, :self.n_components])

    @profile
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the PCA model with the input data and apply dimensionality reduction.

        :param X: The input data, a 2D numpy array of shape (n_samples, n_features).
        :return: The transformed data.
        """
        self.fit(X)
        return self.transform(X)
