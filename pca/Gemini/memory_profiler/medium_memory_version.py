
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
        
from typing import Optional, Tuple

import numpy as np


class PrincipalComponentAnalysis:
    """
    A class implementing Principal Component Analysis (PCA) from scratch using eigen-decomposition.

    This class performs dimensionality reduction on data by finding the principal components,
    which capture the most significant variance in the data.

    Attributes:
        eigenvectors_: (np.ndarray): The eigenvectors of the covariance matrix.
        explained_variance_: (np.ndarray): The explained variance ratio for each principal component.
        n_components_: (int): The number of principal components used.
    """

    @profile
    def __init__(self, n_components: Optional[int] = None) -> None:
        """
        Initializes the PCA object.

        Args:
            n_components: (Optional[int]): The number of principal components to use. Defaults to all.
        """
        self.eigenvectors_: np.ndarray = None
        self.explained_variance_: np.ndarray = None
        self.n_components_: int = n_components

    @profile
    def fit(self, X: np.ndarray) -> None:
        """
        Fits the PCA model to the data.

        This method calculates the covariance matrix, performs eigen-decomposition, and stores
        the eigenvectors and explained variance ratio.

        Args:
            X: (np.ndarray): The data matrix with shape (n_samples, n_features).
        """
        n_samples, n_features = X.shape

        # Center the data
        X_centered = X - np.mean(X, axis=0)

        # Calculate the covariance matrix
        covariance = np.cov(X_centered.T)

        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eig(covariance)

        # Sort eigenvalues and eigenvectors by decreasing order of eigenvalues
        sorted_index = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_index]
        eigenvectors = eigenvectors[:, sorted_index]

        # Determine the number of components to use
        if self.n_components is None:
            self.n_components_ = n_features
        else:
            self.n_components_ = min(self.n_components, n_features)

        # Select the top n_components eigenvectors and explained variance
        self.eigenvectors_ = eigenvectors[:, : self.n_components_]
        self.explained_variance_ = eigenvalues[: self.n_components_] / np.sum(eigenvalues)

    @profile
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transforms the data using the fitted PCA model.

        This method projects the data onto the principal components.

        Args:
            X: (np.ndarray): The data matrix with shape (n_samples, n_features).

        Returns:
            (np.ndarray): The transformed data matrix with shape (n_samples, n_components).
        """
        if self.eigenvectors_ is None:
            raise ValueError("PCA model not fitted. Call fit before transform.")

        # Center the data
        X_centered = X - np.mean(X, axis=0)

        # Project data onto principal components
        return X_centered @ self.eigenvectors_

    @profile
    def fit_transform(self, X: np.ndarray) -> Tuple[np.ndarray, None]:
        """
        Fits the PCA model to the data and then transforms it.

        This method is a convenience function that combines both fit and transform steps.

        Args:
            X: (np.ndarray): The data matrix with shape (n_samples, n_features).

        Returns:
            Tuple[np.ndarray, None]: The transformed data matrix with shape (n_samples, n_components)
                                      and None (for compatibility with scikit-learn).
        """
        self.fit(X)
        return self.transform(X)
