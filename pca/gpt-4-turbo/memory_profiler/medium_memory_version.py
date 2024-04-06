
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

class PrincipalComponentAnalysis:
    """
    Principal Component Analysis (PCA) using Eigenvalue Decomposition.

    PCA is a technique used for dimensionality reduction while preserving as much of the
    data's original variance as possible. It transforms the data into a new coordinate system
    where the greatest variances by some scalar projection of the data come to lie on the
    first coordinate (called the first principal component), the second greatest variance on
    the second coordinate, and so on.
    
    Parameters
    ----------
    n_components : int, optional
        Number of components to keep. If n_components is not set or None,
        all components are kept.

    Attributes
    ----------
    components_ : array, shape (n_components, n_features)
        Principal axes in feature space, representing the directions of
        maximum variance in the data. The components are sorted by
        explained_variance_.
    explained_variance_ : array, shape (n_components,)
        The amount of variance explained by each of the selected components.

    Example
    -------
    >>> from sklearn.datasets import load_iris
    >>> data = load_iris().data
    >>> pca = PrincipalComponentAnalysis(n_components=2)
    >>> pca.fit(data)
    >>> transformed_data = pca.transform(data)
    """
    @profile
    def __init__(self, n_components: int = None) -> None:
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
    
    @profile
    def fit(self, X: np.ndarray) -> None:
        """
        Fit the model with X by computing the mean and principal components.
        This method computes eigenvalues and eigenvectors of the covariance matrix
        of X to find the principal components.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data to perform PCA on.
        """
        # Centering the data (subtract the mean)
        X_centered = X - np.mean(X, axis=0)
        
        # Computing the covariance matrix
        covariance_matrix = np.cov(X_centered, rowvar=False)
        
        # Computing eigenvalues and eigenvectors of the covariance matrix
        eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)
        
        # Sorting the eigenvectors by decreasing eigenvalues
        idx = np.argsort(eigen_values)[::-1]
        eigen_values_sorted = eigen_values[idx]
        eigen_vectors_sorted = eigen_vectors[:, idx]
        
        # Selecting n_components if set, otherwise use all
        if self.n_components is not None:
            self.components_ = eigen_vectors_sorted[:, :self.n_components].T
            self.explained_variance_ = eigen_values_sorted[:self.n_components]
        else:
            self.components_ = eigen_vectors_sorted.T
            self.explained_variance_ = eigen_values_sorted
    
    @profile
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project the data onto the principal axes previously extracted from a training set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
            Transformed data.
        """
        if self.components_ is None:
            raise ValueError("Model has not been fitted yet. Call 'fit' with appropriate data.")
        
        # Centering the data
        X_centered = X - np.mean(X, axis=0)
        
        # Projecting data
        X_projected = np.dot(X_centered, self.components_.T)
        return X_projected
    
    @profile
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data to fit and transform.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
            Transformed data.
        """
        self.fit(X)
        return self.transform(X)
