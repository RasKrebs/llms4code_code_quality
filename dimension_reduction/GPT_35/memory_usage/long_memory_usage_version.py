import numpy as np
import pandas as pd
from memory_profiler import profile


class PrincipalComponentAnalysis:
    """
    Principal Component Analysis (PCA) implementation.
    
    Parameters:
        n_components (int, optional): Number of principal components to keep. Defaults to None (keep all components).
        decomposition_method (str, optional): 'eigen' for eigenvalue decomposition or 'svd' for singular value decomposition. Defaults to 'eigen'.
    
    Attributes:
        n_components (int): Number of principal components to keep.
        decomposition_method (str): Method for decomposition ('eigen' or 'svd').
        eigenvalues (numpy.ndarray): Eigenvalues of the covariance matrix.
        eigenvectors (numpy.ndarray): Eigenvectors (principal components) corresponding to the eigenvalues.
        explained_variance_ratio (numpy.ndarray): Ratio of explained variance for each principal component.
        cumulative_variance_ratio (numpy.ndarray): Cumulative explained variance ratio for each component.
    
    Methods:
        fit(X): Compute eigenvalues and eigenvectors from input data.
        transform(X, n_components=None): Project input data onto principal components.
        fit_transform(X, n_components=None): Fit the model and transform input data.
    """
    @profile
    def __init__(self, n_components=None, decomposition_method='eigen'):
        self.n_components = n_components
        self.decomposition_method = decomposition_method


    @profile
    def fit(self, X):
        """
        Fit the PCA model on input data and compute eigenvalues, eigenvectors, explained variance ratio, and cumulative variance ratio.

        Parameters:
            X (numpy.ndarray, pandas.DataFrame, or list of lists): Input data for PCA.
        
        Returns:
            None
        """
        # Ensure X is a numpy array
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        elif isinstance(X, list):
            X = np.array(X)

        # Center the data by subtracting the mean
        mean = np.mean(X, axis=0)
        centered_data = X - mean

        if self.decomposition_method == 'eigen':
            # Compute the covariance matrix
            covariance_matrix = np.cov(centered_data, rowvar=False)

            # Perform eigenvalue decomposition
            eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        elif self.decomposition_method == 'svd':
            # Perform Singular Value Decomposition (SVD)
            _, S, Vt = np.linalg.svd(centered_data, full_matrices=False)
            eigenvalues = S ** 2
            eigenvectors = Vt.T

        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Store the eigenvalues and eigenvectors
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors

        # Compute explained variance ratio and cumulative variance ratio
        total_variance = np.sum(eigenvalues)
        explained_variance_ratio = eigenvalues / total_variance
        self.explained_variance_ratio = explained_variance_ratio

        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        self.cumulative_variance_ratio = cumulative_variance_ratio

    @profile
    def transform(self, X, n_components=None):
        """
        Project input data onto principal components.

        Parameters:
            X (numpy.ndarray, pandas.DataFrame, or list of lists): Input data for transformation.
            n_components (int, optional): Number of principal components to keep. Defaults to None (keep all components).

        Returns:
            numpy.ndarray: Transformed data projected onto the specified number of components.
        """
        if n_components is None:
            n_components = self.n_components

        # Ensure X is a numpy array
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        elif isinstance(X, list):
            X = np.array(X)

        if n_components is None or n_components >= X.shape[1]:
            # Use all components
            return X.dot(self.eigenvectors)
        else:
            # Use a subset of components
            return X.dot(self.eigenvectors[:, :n_components])
    
    # @profile has been omitted from this method as its just using the other two methods        
    def fit_transform(self, X, n_components=None):
        """
        Fit the model and transform input data in a single step.

        Parameters:
            X (numpy.ndarray, pandas.DataFrame, or list of lists): Input data for PCA.
            n_components (int, optional): Number of principal components to keep. Defaults to None (keep all components).

        Returns:
            numpy.ndarray: Transformed data projected onto the specified number of components.
        """
        self.fit(X)
        return self.transform(X, n_components)

# Execute script data to get memory usage
if __name__ == "__main__":
    # For reproducibility
    np.random.seed(42)

    # Generate random data
    data = np.random.rand(10000, 100)
    
    # Create PCA object
    pca = PrincipalComponentAnalysis()

    # Fit and transform data
    _ = pca.fit_transform(data)