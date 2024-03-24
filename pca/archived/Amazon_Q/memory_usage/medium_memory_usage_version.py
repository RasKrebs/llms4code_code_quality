from memory_profiler import profile
from typing import Optional

import numpy as np

class PrincipalComponentAnalysis:
    """
    Implements Principal Component Analysis algorithm for dimensionality reduction.

    Performs PCA using either the eigenvalue decomposition or SVD method.
    """
    @profile
    def __init__(self, n_components: Optional[int] = None, 
                 decomposition_method: str = 'eigen'):
        """
        n_components: Number of components to keep. If None, keep all.
        decomposition_method: 'eigen' to use eigenvalue decomposition, 
                              'svd' to use SVD decomposition.
        """
        self.n_components = n_components
        self.decomposition_method = decomposition_method
    @profile
    def fit(self, X):
          """
          Fit the model with X.
    
          X: Array-like of shape (n_samples, n_features)
          """
    
          # Centering data
          self.mean_ = np.mean(X, axis=0)  
          X_centered = X - self.mean_
    
          if self.decomposition_method == 'eigen':
             # Eigendecomposition of covariance matrix
             cov_mat = np.cov(X_centered.T)
             s      else:
             # SVD approach  
             u, s, v = np.linalg.svd(X_centered)
             self.eigenvalues_ = s**2
             self.eigenvectors_ = v
    
          return self
    @profile
    def transform(self, X):
        """
        Apply dimensionality reduction to X.  

        X: Array-like of shape (n_samples, n_features)  
        """

        X_transformed = np.dot(X - self.mean_, self.eigenvectors_[:, :self.n_components])
        return X_transformed
    @profile
    def fit_transform(self, X):
        """
        Fit to data, then transform it.

        X: Array-like of shape (n_samples, n_features)
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
    pca = PrincipalComponentAnalysis()

    # Fit and transform data
    _ = pca.fit_transform(data)