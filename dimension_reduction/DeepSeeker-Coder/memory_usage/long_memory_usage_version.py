from memory_profiler import profile
import numpy as np
import pandas as pd
from typing import Union, List, Optional

class PrincipalComponentAnalysis:
    @profile
    def __init__(self, n_components: Optional[int] = None, decomposition_method: str = 'eigen'):
        self.n_components = n_components
        self.decomposition_method = decomposition_method
        self.eigen_values = None
        self.eigen_vectors = None
        self.explained_variance_ratio = None
        self.cumulative_explained_variance_ratio = None

    @profile
    def fit(self, X: Union[np.ndarray, pd.DataFrame, List[List]]) -> None:
        """
        Compute the eigen values and eigen vectors for the inputted data.
        If n_components is specified, only store top n_components eigen values and eigen vectors.
        Also compute the explained variance ratio and cumulative sum of explained variance ratio for each component.
        """
        if isinstance(X, list):
            X = np.array(X)
        elif isinstance(X, pd.DataFrame):
            X = X.values

        # Center the data
        X = X - np.mean(X, axis=0)

        # Compute the covariance matrix
        cov_matrix = np.cov(X.T)

        # Compute the eigen values and eigen vectors
        if self.decomposition_method == 'eigen':
            eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
        elif self.decomposition_method == 'svd':
            _, eigen_values, eigen_vectors = np.linalg.svd(cov_matrix)

        # Sort the eigen values and eigen vectors in descending order
        idx = eigen_values.argsort()[::-1]
        eigen_values = eigen_values[idx]
        eigen_vectors = eigen_vectors[:, idx]

        # Store the eigen values and eigen vectors
        if self.n_components is not None:
            self.eigen_values = eigen_values[:self.n_components]
            self.eigen_vectors = eigen_vectors[:, :self.n_components]
        else:
            self.eigen_values = eigen_values
            self.eigen_vectors = eigen_vectors

        # Compute the explained variance ratio
        self.explained_variance_ratio = self.eigen_values / np.sum(self.eigen_values)

        # Compute the cumulative sum of explained variance ratio
        self.cumulative_explained_variance_ratio = np.cumsum(self.explained_variance_ratio)

    @profile
    def transform(self, X: Union[np.ndarray, pd.DataFrame, List[List]], n_components: Optional[int] = None) -> np.ndarray:
        """
        Project the inputted data onto the components.
        If n_components is specified, project onto the top n_components components.
        """
        if isinstance(X, list):
            X = np.array(X)
        elif isinstance(X, pd.DataFrame):
            X = X.values

        # Center the data
        X = X - np.mean(X, axis=0)

        # Project the data onto the components
        if n_components is not None:
            return np.dot(X, self.eigen_vectors[:, :n_components])
        else:
            return np.dot(X, self.eigen_vectors)

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame, List[List]], n_components: Optional[int] = None) -> np.ndarray:
        """
        Call the fit method and then the transform method.
        If n_components is specified, use it for both methods.
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
    pca.fit(data)
    _ = pca.transform(data)