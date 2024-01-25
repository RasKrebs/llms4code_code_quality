import numpy as np
from typing import Union, Optional

class PrincipalComponentAnalysis():
    """
    A class that implements Principal Component Analysis algorithm.

    Attributes:
        n_components (int): Number of components to use. If not specified it will default to all.
        decomposition_method (str): Method of decomposition, either 'eigen' or 'svd'.
            If not specified it will default to 'eigen'.

    """

    def __init__(self, n_components: Optional[int] = None,
                 decomposition_method: str = "eigen"):
        self.n_components = n_components
        self.decomposition_method = decomposition_method

    def fit(self, X: np.ndarray) -> None:
        """Fit PCA model to data.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            None. Updates the mean, covariance matrix and eigenvalues/vectors.

        """
        # Mean centering
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        n_samples = X.shape[0]

        if self.decomposition_method == 'eigen':
            # Computing covariance matrix
            cov_mat = np.cov(X_centered, rowvar=False)

            # Eigen decomposition
            eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)

            # Sort eigenvalues and vectors in descending order
            sorted_index = np.argsort(eigen_values)[::-1]
            self.sorted_eigenvalues = eigen_values[sorted_index]
            self.sorted_eigenvectors = eigen_vectors[:, sorted_index]
        elif self.decomposition_method == 'svd':
            # Singular Value Decomposition
            U, Sigma, VT = np.linalg.svd(X_centered)

            # Retain only the first k eigenvectors (sorted in decreasing order of their corresponding eigenvalues)
            self.sorted_eigenvalues = Sigma**2 / n_samples
            self.sorted_eigenvectors = U
        else:
            raise ValueError("Invalid decomposition method. Choose 'eigen' or 'svd'.")

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply dimensionality reduction on input data.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            Transformed data with reduced dimensionality.

        """
        X -= self.mean

        if self.n_components is None:
            return np.dot(X, self.sorted_eigenvectors)
        else:
            return np.dot(X, self.sorted_eigenvectors[:, :self.n_components])

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit PCA model to data and then apply dimensionality reduction on it.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            Transformed data with reduced dimensionality.

        """
        self.fit(X)
        return self.transform(X)