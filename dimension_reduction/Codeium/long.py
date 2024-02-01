import numpy as np
import pandas as pd

class PrincipalComponentAnalysis:
    """
    Principal Component Analysis (PCA) algorithm implementation from scratch.
    """

    def __init__(self, n_components=None, decomposition_method='eigen'):
        """
        Initialize the PCA object.

        Args:
        - n_components (int): Number of components to keep. If not specified, it defaults to all.
        - decomposition_method (str): Method for matrix decomposition, either 'eigen' or 'svd'. Defaults to 'eigen'.
        """
        self.n_components = n_components
        self.decomposition_method = decomposition_method
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.cumulative_explained_variance_ratio_ = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame, List[List[float]]]):
        """
        Compute the eigenvalues and eigenvectors for the input data and store them on the object.

        Args:
        - X (array-like): Input data as a numpy array, pandas dataframe, or a list of lists.
        """
        # Compute the covariance matrix
        cov_matrix = np.cov(X, rowvar=False)

        # Perform matrix decomposition based on the specified method
        if self.decomposition_method == 'eigen':
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        elif self.decomposition_method == 'svd':
            _, _, eigenvectors = np.linalg.svd(X)
            eigenvalues = np.square(eigenvectors)
        else:
            raise ValueError("Invalid decomposition method. Please use 'eigen' or 'svd'.")

        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Store components and explained variance
        self.components_ = eigenvectors[:, :self.n_components]
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = eigenvalues / np.sum(eigenvalues)
        self.cumulative_explained_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)

    def transform(self, X: Union[np.ndarray, pd.DataFrame, List[List[float]]], n_components=None):
        """
        Project the input data onto the components and return the projected data.

        Args:
        - X (array-like): Input data as a numpy array, pandas dataframe, or a list of lists.
        - n_components (int): Number of components to project onto. If not specified, it defaults to all.

        Returns:
        - array-like: Projected data.
        """
        if n_components is None:
            n_components = self.n_components

        projected_data = np.dot(X, self.components_[:, :n_components])
        return projected_data

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame, List[List[float]]], n_components=None):
        """
        Compute the eigenvalues and eigenvectors for the input data, and then project the data onto the components.

        Args:
        - X (array-like): Input data as a numpy array, pandas dataframe, or a list of lists.
        - n_components (int): Number of components to project onto. If not specified, it defaults to all.

        Returns:
        - array-like: Projected data.
        """
        self.fit(X)
        return self.transform(X, n_components)