
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
    A class to perform Principal Component Analysis using eigen value decomposition.

    Attributes:
    n_components : int, optional
        Number of components to use. Default is all.

    Methods:
    fit(X: np.ndarray) -> None:
        Fit the PCA model with the data.
    
    transform(X: np.ndarray) -> np.ndarray:
        Apply dimensionality reduction to X.
    
    fit_transform(X: np.ndarray) -> np.ndarray:
        Fit the PCA model with the data and apply dimensionality reduction to X.
    """

    @profile
    def __init__(self, n_components=None):
        self.n_components = n_components

    @profile
    def fit(self, X: np.ndarray) -> None:
        """
        Fit the PCA model with the data.
        
        Parameters:
        X : np.ndarray
            Input data of shape (n_samples, n_features).
        """
        # Centering the data
        mean = np.mean(X, axis=0)
        X_centered = X - mean
        
        # Calculate the covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)
        
        # Perform eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Sort eigenvalues and eigenvectors in decreasing order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Select number of components
        if self.n_components is not None:
            self.components = eigenvectors[:, :self.n_components]
        else:
            self.components = eigenvectors

    @profile
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply dimensionality reduction to X.
        
        Parameters:
        X : np.ndarray
            Input data of shape (n_samples, n_features).
        
        Returns:
        np.ndarray
            Transformed data of shape (n_samples, n_components).
        """
        X_transformed = np.dot(X, self.components)
        return X_transformed

    @profile
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the PCA model with the data and apply dimensionality reduction to X.
        
        Parameters:
        X : np.ndarray
            Input data of shape (n_samples, n_features).
        
        Returns:
        np.ndarray
            Transformed data of shape (n_samples, n_components).
        """
        self.fit(X)
        X_transformed = self.transform(X)
        return X_transformed
