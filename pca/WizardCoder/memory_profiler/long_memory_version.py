
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
from typing import Union, Optional

class PCA:
    @profile
    def __init__(self, n_components: Optional[int] = None):
        self.n_components = n_components
        self.eigenvalues = []
        self.explained_variance_ratio = []
        self.cumulative_explained_variance_ratio = []

    @profile
    def fit(self, X: Union[np.ndarray, list]) -> None:
        """Compute PCA components for the data"""

        # Convert input to numpy array if it's a list
        if isinstance(X, list):
            X = np.array(X)

        # Normalize data by subtracting mean and dividing by standard deviation
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        X = (X - self.mean_) / self.std_

        # Get the covariance matrix
        cov_matrix = np.cov(X.T)

        # Compute eigenvalues and eigenvectors
        self.eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Sort eigenvalues and eigenvectors in descending order of the eigenvalues
        idx = self.eigenvalues.argsort()[::-1]
        self.eigenvalues = self.eigenvalues[idx]
        eigenvectors = eigenvectors[:,idx]

        # Calculate explained variance ratio and cumulative explained variance ratio
        self.explained_variance_ratio = (self.eigenvalues/sum(self.eigenvalues))*100
        self.cumulative_explained_variance_ratio = np.cumsum(self.explained_variance_ratio)

    @profile
    def transform(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """Transform the data"""

        # Convert input to numpy array if it's a list
        if isinstance(X, list):
            X = np.array(X)

        # Normalize data by subtracting mean and dividing by standard deviation
        X = (X - self.mean_) / self.std_

        # Project onto the first n_components eigenvectors
        if self.n_components is not None:
            X = X @ self.eigenvectors[:, :self.n_components]
        else:
            X = X @ self.eigenvectors

        return X

    @profile
    def fit_transform(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """Fit and transform the data"""

        self.fit(X)
        return self.transform(X)
