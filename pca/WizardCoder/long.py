import numpy as np
from typing import Union, List
from pandas import DataFrame

class PrincipalComponentAnalysis:
    """Principal Component Analysis from scratch."""

    def __init__(self, n_components: int=None, decomposition_method: str='eigen') -> None:
        self.n_components = n_components
        self.decomposition_method = decomposition_method

    def fit(self, X: Union[np.ndarray, List[List[float]], DataFrame]) -> 'PrincipalComponentAnalysis':
        """Compute the eigen values and vectors for the inputted data."""

        # Convert to numpy array if not already
        if isinstance(X, (DataFrame, list)):
            X = np.array(X)

        # Subtract mean from data
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_

        # Compute covariance matrix
        C = np.cov(X.T)

        if self.decomposition_method == 'eigen':
            # Calculate Eigenvalues and Eigenvectors
            eigenvalues, eigenvectors = np.linalg.eig(C)

            # Sort eigenvalues and eigenvectors in descending order
            indices = np.argsort(eigenvalues)[::-1]
        elif self.decomposition_method == 'svd':
            U, S, VT = np.linalg.svd(X, full_matrices=False)
            eigenvectors, eigenvalues = VT.T, (S**2) / (X.shape[0] - 1)
        else:
            raise ValueError("decomposition method must be either 'eigen' or 'svd'")

        # Compute explained variance and cumulative sum of explained variance
        total_var = np.sum(eigenvalues)
        self.explained_variance_ratio_ = eigenvalues / total_var
        self.cumulative_explained_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)

        # Get the top n components if specified
        if self.n_components is not None:
            eigenvectors = eigenvectors[indices[:self.n_components]]

        self.eigenvalues_ = eigenvalues
        self.components_ = eigenvectors

        return self

    def transform(self, X: Union[np.ndarray, List[List[float]], DataFrame], n_components:
int=None) -> np.ndarray:
        """Project the inputted data onto the components."""
        if isinstance(X, (DataFrame, list)):
            X = np.array(X)

        # Subtract mean from data
        X -= self.mean_

        # Project data onto components
        if n_components is None:
            return X @ self.components_.T
        else:
            return X @ self.components_[:n_components].T

    def fit_transform(self, X: Union[np.ndarray, List[List[float]], DataFrame], n_components:
int=None) -> np.ndarray:
        """Compute the eigen values and vectors for the inputted data and project the data onto
the components."""
        self.fit(X)
        return self.transform(X, n_components)