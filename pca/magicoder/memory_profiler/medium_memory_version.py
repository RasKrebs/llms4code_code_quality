
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
from typing import Optional
from scipy.linalg import eigh

class PrincipalComponentAnalysis:
    """Class for principal component analysis"""

    @profile
    def __init__(self, n_components:Optional[int]=None):
        self.n_components = n_components

    @profile
    def fit(self, X:np.ndarray) -> None:
        """Fit the PCA model with input data"""
        # Subtract mean of each feature in dataset to get a zero-mean dataset
        self.X = X - np.mean(X, axis=0)

    @profile
    def transform(self, X:np.ndarray) -> np.ndarray:
        """Transform input data using fitted model"""
        # Compute covariance matrix of the centered data
        cov_mat = np.cov(self.X, rowvar=False)

        # Eigendecomposition of the covariance matrix to get eigenvectors and eigenvalues
        eigval, self.eigvec = eigh(cov_mat)

        # Sort eigenpairs by eigenvalue in descending order
        idx = np.argsort(eigval)[::-1]
        self.eigval = eigval[idx]
        self.eigvec = self.eigvec[:,idx]

        if self.n_components is not None:
            # Select the first n_components eigenvectors (principal components)
            self.eigenvectors = self.eigvec[0:self.n_components, :]

        else:
            self.eigenvectors = self.eigvec

        # Transform the data onto the new feature space
        return np.dot(X - np.mean(X), self.eigenvectors.T).T

    @profile
    def fit_transform(self, X:np.ndarray) -> np.ndarray:
        """Fit and transform input data"""
        self.fit(X)
        return self.transform(X)
