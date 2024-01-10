import numpy as np
import pandas as pd

class PrincipalComponentAnalysis:
    """
    Principal Component Analysis (PCA) class for dimensionality reduction.

    Parameters:
    ----------
    n_components : int, optional
        Number of principal components to retain. Defaults to None, which keeps all components.
    """

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance_ratio_ = None
        self.cumulative_variance_ratio_ = None

    def fit(self, X):
        """
        Fit the PCA model to the input data and compute the eigenvalues, eigenvectors,
        explained variance ratio, and cumulative explained variance ratio.

        Parameters:
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data for PCA. Can be a NumPy array, Pandas DataFrame, or a list of lists.
        """

        # Ensure X is a NumPy array
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        elif not isinstance(X, np.ndarray):
            X = np.array(X)

        # Compute the mean of the data
        self.mean = np.mean(X, axis=0)

        # Center the data
        centered_data = X - self.mean

        # Compute the covariance matrix
        cov_matrix = np.cov(centered_data, rowvar=False)

        # Compute the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Store all or n_components eigenvectors as components
        if self.n_components is None:
            self.components = eigenvectors
        else:
            self.components = eigenvectors[:, :self.n_components]

        # Compute explained variance ratio and cumulative explained variance ratio
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = eigenvalues / total_variance
        self.cumulative_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)

    def transform(self, X, n_components=None):
        """
        Project the input data onto the principal components.

        Parameters:
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data for projection. Can be a NumPy array, Pandas DataFrame, or a list of lists.
        n_components : int, optional
            Number of principal components to project onto. Defaults to None, which uses all components.

        Returns:
        -------
        transformed_data : array-like, shape (n_samples, n_components)
            Projected data.
        """
        if n_components is None:
            n_components = self.n_components

        # Ensure X is a NumPy array
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        elif not isinstance(X, np.ndarray):
            X = np.array(X)

        # Center the data
        centered_data = X - self.mean

        # Project the data onto the specified number of components
        if n_components is None:
            return np.dot(centered_data, self.components)
        else:
            return np.dot(centered_data, self.components[:, :n_components])

    def fit_transform(self, X, n_components=None):
        """
        Fit the PCA model to the input data and then project the data onto the principal components.

        Parameters:
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data for PCA and projection. Can be a NumPy array, Pandas DataFrame, or a list of lists.
        n_components : int, optional
            Number of principal components to retain. Defaults to None, which uses all components.

        Returns:
        -------
        transformed_data : array-like, shape (n_samples, n_components)
            Projected data.
        """
        self.fit(X)
        return self.transform(X, n_components)