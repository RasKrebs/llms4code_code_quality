from memory_profiler import profile
import numpy as np
from typing import Optional, Union

class PrincipalComponentAnalysis:
    """
    Principal Component Analysis (PCA) class.

    Parameters
    ----------
    n_components : int, optional
        Number of components to use. If not specified, all components are used.
    decomposition_method : str, optional
        Method to use for decomposition. Either 'eigen' or 'svd'. If not specified, 'eigen' is used.
    """

    @profile
    def __init__(self, n_components: Optional[int] = None, decomposition_method: Optional[str] = 'eigen'):
        self.n_components = n_components
        self.decomposition_method = decomposition_method
        self.components = None
        self.mean = None

    @profile
    def fit(self, X: np.ndarray) -> None:
        """
        Fit the model with X.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            Training data.
        """
        # Calculate the mean
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # Calculate the covariance matrix
        cov = np.cov(X.T)

        # Eigen decomposition
        if self.decomposition_method == 'eigen':
            eigenvalues, eigenvectors = np.linalg.eig(cov)
            # Sort eigenvectors
            eigenvectors = eigenvectors.T
            idxs = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idxs]
            eigenvectors = eigenvectors[idxs]
        # Singular Value Decomposition
        elif self.decomposition_method == 'svd':
            eigenvectors, s, v = np.linalg.svd(X.T)
            eigenvalues = s**2 / (X.shape[0] - 1)

        # Store first n eigenvectors
        self.components = eigenvectors[0:self.n_components]

    @profile
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the dimensionality reduction on X.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        X_new : numpy array of shape (n_samples, n_components)
            Transformed values.
        """
        X = X - self.mean
        return np.dot(X, self.components.T)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        X_new : numpy array of shape (n_samples, n_components)
            Transformed values.
        """
        self.fit(X)
        return self.transform(X)
    
# Execute script data to get memory usage
if __name__ == "__main__":
    # For reproducibility
    np.random.seed(42)

    # Generate random data
    data = np.random.rand(10000, 100)
    
    # Create PCA object
    pca = PrincipalComponentAnalysis(n_components=5)

    # Fit and transform data
    _ = pca.fit_transform(data)