
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

class CustomPCA:
    @profile
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    @profile
    def fit(self, X):
        # Compute the mean of the data
        self.mean = np.mean(X, axis=0)
        
        # Center the data
        X_centered = X - self.mean
        
        # Compute the covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)
        
        # Perform eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Sort the eigenvectors based on eigenvalues
        idx = np.argsort(eigenvalues)[::-1]
        self.components = eigenvectors[:, idx][:, :self.n_components]

    @profile
    def transform(self, X):
        # Center the data
        X_centered = X - self.mean
        
        # Project the data onto the principal components
        return np.dot(X_centered, self.components)

    @profile
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
