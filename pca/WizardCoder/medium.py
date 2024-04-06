import numpy as np
from sklearn import datasets
from typing import Union

class PrincipalComponentAnalysis:
    def __init__(self, n_components: int = None) -> None:
        """Initialize the PCA object with number of components to be retained.

        Args:
            n_components (int): Number of components to retain. Defaults to all components.

        Returns:
            None
        """
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit(self, X: np.ndarray) -> 'PrincipalComponentAnalysis':
        """Fit the PCA model with X by calculating mean and components.

        Args:
            X (np.ndarray): Data to fit the model.

        Returns:
            self (PrincipalComponentAnalysis): The fitted PrincipalComponentAnalysis object.
        """

        # Calculate the mean of each feature
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # Calculating the covariance matrix of the mean-centered data.
        cov_mat = np.cov(X.T)

        # Compute eigen values and eigen vectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_mat)

        # Sort eigenvalues in descending order and sort the corresponding eigenvectors
        sorted_index = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_index]
        sorted_eigenvectors = eigenvectors[:,sorted_index]

        # Selecting the first n eigenvectors as per given dimension
        if self.n_components is not None:
            sorted_eigenvectors = sorted_eigenvectors[:, :self.n_components]

        # Store the resultant vectors
        self.components = sorted_eigenvectors

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the given data using the fitted PCA model.

        Args:
            X (np.ndarray): Data to be transformed.

        Returns:
            X_transformed (np.ndarray): Transformed data.
        """

        # Mean centering
        X = X - self.mean

        # Return the projection of X onto the principal components
        return np.dot(X, self.components)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform the data using the fitted PCA model.

        Args:
            X (np.ndarray): Data to be transformed.

        Returns:
            X_transformed (np.ndarray): Transformed data.
        """
        return self.fit(X).transform(X)
