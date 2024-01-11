import numpy as np
import pandas as pd
from memory_profiler import profile
from typing import Union, Optional, List

class PrincipalComponentAnalysis:
    """
    Principal Component Analysis (PCA) implementation.

    Parameters:
    n_components: int, optional
        Number of components to keep. If n_components is not set or None,
        all components are kept.
    decomposition_method: str, optional
        The method used for decomposition. Can be either 'eigen' or 'svd'.
        Defaults to 'eigen'.

    Attributes:
    components_: array, shape (n_components, n_features)
        Principal axes in feature space.
    explained_variance_: array, shape (n_components,)
        The amount of variance explained by each of the selected components.
    explained_variance_ratio_: array, shape (n_components,)
        Percentage of variance explained by each of the selected components.
    cumulative_explained_variance_ratio_: array, shape (n_components,)
        Cumulative sum of explained variance ratio.
    """
    @profile
    def __init__(self, n_components: Optional[int] = None, decomposition_method: str = 'eigen'):
        self.n_components = n_components
        self.decomposition_method = decomposition_method
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.cumulative_explained_variance_ratio_ = None

    @profile
    def fit(self, X: Union[np.ndarray, pd.DataFrame, List[List[float]]]):
        """
        Fit the model with X.

        Parameters:
        X: array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        """
        if isinstance(X, list):
            X = np.array(X)
        elif isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        # Standardize data
        X = X - np.mean(X, axis=0)

        # Compute covariance matrix
        cov_matrix = np.cov(X, rowvar=False)

        # Eigenvalue decomposition or SVD
        if self.decomposition_method == 'eigen':
            eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)
        elif self.decomposition_method == 'svd':
            U, S, V = np.linalg.svd(X.T)
            eigen_values = S ** 2 / (len(X) - 1)
            eigen_vectors = U
        else:
            raise ValueError("Decomposition method must be either 'eigen' or 'svd'.")

        # Sort eigenvalues and eigenvectors
        idx = np.argsort(eigen_values)[::-1]
        eigen_values = eigen_values[idx]
        eigen_vectors = eigen_vectors[:, idx]

        # Select n_components
        if self.n_components is not None:
            eigen_values = eigen_values[:self.n_components]
            eigen_vectors = eigen_vectors[:, :self.n_components]

        self.components_ = eigen_vectors.T
        self.explained_variance_ = eigen_values
        self.explained_variance_ratio_ = eigen_values / eigen_values.sum()
        self.cumulative_explained_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)

    @profile
    def transform(self, X: Union[np.ndarray, pd.DataFrame, List[List[float]]], n_components: Optional[int] = None):
        """
        Apply dimensionality reduction to X.

        X is projected on the first principal components previously extracted
        from a training set.

        Parameters:
        X: array-like, shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.
        n_components: int, optional
            Number of components to project onto. If not specified, use all.

        Returns:
        X_new: array-like, shape (n_samples, n_components)
            Transformed values.
        """
        if isinstance(X, list):
            X = np.array(X)
        elif isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        X = X - np.mean(X, axis=0)

        if n_components is None:
            n_components = self.n_components

        X_transformed = np.dot(X, self.components_[:n_components].T)
        return X_transformed

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame, List[List[float]]], n_components: Optional[int] = None):
        """
        Fit the model with X and apply the dimensionality reduction on X.

        Parameters:
        X: array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        n_components: int, optional
            Number of components to project onto. If not specified, use all.

        Returns:
        X_new: array-like, shape (n_samples, n_components)
            Transformed values.
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