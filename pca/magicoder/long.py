import numpy as np
from typing import Union, Tuple

class PCA:
    """Principal Component Analysis"""

    def __init__(self, n_components: int=None):
        self.n_components = n_components  # number of components to use

    def fit(self, X: Union[np.ndarray, list]) -> None:
        """Fit the data."""
        if isinstance(X, list):
            X = np.array(X)

        self._mean = np.mean(X, axis=0)  # calculate mean for subtraction of mean value

        X_std = X - self._mean  # subtract mean from each dimension to center the data

        cov_mat = np.cov(X_std, rowvar=False)
        eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)

        sorted_indexes = np.argsort(eigen_values)[::-1]  # sort eigenvalues in decreasing order
        self._sorted_eigenvalue = eigen_values[sorted_indexes]
        self._sorted_eigenvectors = eigen_vectors[:, sorted_indexes]

        total = sum(self._sorted_eigenvalue)  # Total of all eigenvalues

        if self.n_components:
            self.explained_variance_ratio = [(i / total)*100 for i in self._sorted_eigenvalue[:self.n_components]]
        else:
            self.explained_variance_ratio = 100 * (self._sorted_eigenvalue/total)  # all eigenvalues

    def transform(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """Project the input data onto the components."""
        if isinstance(X, list):
            X = np.array(X)

        X_std = X - self._mean  # subtract mean from each dimension to center the data

        return np.dot(self._sorted_eigenvectors[:, :self.n_components], X_std.T).T

    def fit_transform(self, X: Union[np.ndarray, list]) -> Tuple[Union[float, list, np.ndarray], np.ndarray]:
        """Fit the data and project it."""
        self.fit(X)

        return (self._sorted_eigenvalue[:self.n_components],  # eigenvalues of first n components
                self.transform(X))