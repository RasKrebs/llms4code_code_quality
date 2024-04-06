
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
    """
    Principal Component Analysis (PCA)

    Parameters
    ----------
    n_components : int, optional (default=None)
        Number of components to keep.
        If n_components is not set all components are kept.

    Attributes
    ----------
    components_ : array, [n_components, n_features]
        Principal axes in feature space, representing the directions of
        maximum variance in the data.

    explained_variance_ : array, [n_components]
        The amount of variance explained by each of the selected components.

    explained_variance_ratio_ : array, [n_components]
        Percentage of variance explained by each of the selected components.

    cumulative_explained_variance_ratio_ : array, [n_components]
        Cumulative sum of explained variance ratio.
    """

    @profile
    def __init__(self, n_components: Optional[int] = None):
        self.n_components = n_components

    @profile
    def fit(self, X: Union[np.ndarray, list]) -> None:
        """
        Fit the model with X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        """
        X = np.array(X)
        n_samples, n_features = X.shape

        # Center data
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_

        # Compute covariance matrix
        cov_matrix = np.cov(X.T)

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Sort eigenvalues and corresponding eigenvectors
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Determine number of components
        if self.n_components is None:
            self.n_components = n_features

        # Store results
        self.components_ = eigenvectors[:, :self.n_components]
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = self.explained_variance_ / np.sum(eigenvalues)
        self.cumulative_explained_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)

    @profile
    def transform(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """
        Apply dimensionality reduction on X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_transformed : array, shape (n_samples, n_components)
            Projection of X in the first principal components space.
        """
        X = np.array(X)
        X -= self.mean_
        return np.dot(X, self.components_)

    @profile
    def fit_transform(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """
        Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_transformed : array, shape (n_samples, n_components)
            Projection of X in the first principal components space.
        """
        self.fit(X)
        return self.transform(X)
