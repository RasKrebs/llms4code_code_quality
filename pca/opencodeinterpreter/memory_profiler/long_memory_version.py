
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
from typing import Union, List

class PCA:
    @profile
    def __init__(self, n_components: int = None):
        self.n_components = n_components
        self.eigenvalues = None
        self.explained_variance_ratio = None
        self.cumulative_explained_variance = None
        self.mean = None
        self.components = None

    @profile
    def fit(self, X: Union[List[float], np.ndarray]) -> 'PCA':

        # Convert input to numpy array if necessary
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)

        # Calculate mean of each feature
        self.mean = np.mean(X, axis=0)

        # Subtract the mean from each data point
        X -= self.mean

        # Compute covariance matrix
        covariance_matrix = (1/(X.shape[0] - 1)) * np.dot(X.T, X)

        # Calculate eigen values and eigen vectors
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # Sort eigenvalues in descending order and sort corresponding eigen vectors accordingly
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Keep only the first n components
        if self.n_components is None:
            self.n_components = X.shape[1]

        self.explained_variance_ratio = (self.eigenvalues[:self.n_components]) / np.sum(self.eigenvalues)

        # Calculate cumulative sum of explained variance ratio
        self.cumulative_explained_variance = np.cumsum(self.explained_variance_ratio)

        # Normalize eigenvectors for unit vectors and keep only the required components
        self.components = (eigenvectors[:, :self.n_components].T /
                             np.sqrt((eigenvectors[:, :self.n_components]**2).sum(axis=0))).T

        return self

    @profile
    def transform(self, X: Union[List[float], np.ndarray]) -> np.ndarray:

        # Convert input to numpy array if necessary
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)

        # Subtract the mean from each data point
        X -= self.mean

        # Project data onto components
        projected_data = np.dot(X, self.components.T)

        return projected_data

    @profile
    def fit_transform(self, X: Union[List[float], np.ndarray]) -> np.ndarray:

        # Fit PCA on the input data
        self = self.fit(X)

        # Transform the input data
        transformed_data = self.transform(X)

        return transformed_data
