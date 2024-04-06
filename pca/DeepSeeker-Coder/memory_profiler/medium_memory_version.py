
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
from typing import Union, Tuple

class PrincipalComponentAnalysis:
    """
    A class for performing Principal Component Analysis (PCA) on a dataset.

    Attributes:
    -----------
    n_components : int
        Number of principal components to keep.
    components : np.ndarray
        Principal components in feature space, representing the directions of maximum variance in the data.
    explained_variance : np.ndarray
        The amount of variance explained by each of the selected components.
    mean : np.ndarray
        Per-feature empirical mean, estimated from the training set.

    Methods:
    --------
    fit(X: np.ndarray) -> None:
        Computes the principal components and the explained variance.
    transform(X: np.ndarray) -> np.ndarray:
        Transforms the data using the computed principal components.
    fit_transform(X: np.ndarray) -> np.ndarray:
        Fits the model with X and applies the dimensionality reduction on X.
    """

    @profile
    def __init__(self, n_components: Union[int, None] = None):
        """
        Parameters:
        -----------
        n_components : int, optional
            Number of components to keep. If n_components is not set all components are kept.
        """
        self.n_components = n_components
        self.components = None
        self.explained_variance = None
        self.mean = None

    @profile
    def fit(self, X: np.ndarray) -> None:
        """
        Fit the model with X.

        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and n_features is the number of features.
        """
        # Center data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Compute covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort eigenvalues and corresponding eigenvectors
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Store explained variance
        self.explained_variance = eigenvalues

        # Store components
        self.components = eigenvectors.T

        # Reduce dimensions if n_components is set
        if self.n_components is not None:
            self.components = self.components[:self.n_components]
            self.explained_variance = self.explained_variance[:self.n_components]

    @profile
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply dimensionality reduction to X.

        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Data to transform.

        Returns:
        --------
        X_transformed : np.ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        # Center data
        X_centered = X - self.mean

        # Apply transformation
        X_transformed = np.dot(X_centered, self.components.T)

        return X_transformed

    @profile
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the model with X and apply the dimensionality reduction on X.

        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and n_features is the number of features.

        Returns:
        --------
        X_transformed : np.ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        self.fit(X)
        return self.transform(X)
