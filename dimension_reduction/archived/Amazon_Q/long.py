from typing import List, Union
import numpy as np
from sklearn.decomposition import PCA

class PCA:
    def __init__(self, n_components=None, method='svd'):
        self.n_components = n_components
        self.method = method
        self.components_ = None

    def fit(self, X):

        # Center data
        mean = np.mean(X, axis=0)
        X_centered = X - mean

        if self.method == 'svd':
          # SVD method  
          cov_mat = np.cov(X_centered, rowvar=False)
          U, S, Vt = np.linalg.svd(cov_mat)

        elif self.method == 'eigen':
          # Eigenvalue decomposition method
          cov_mat = np.cov(X_centered, rowvar=False)  
          eigenvalues, eigenvectors = np.linalg.eig(cov_mat)
          Vt = eigenvectors

        # Select components
        self.components_ = Vt[:,:self.n_components]
        self.explained_variance_ = self.pca.explained_variance_
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
    
    def transform(self, X):
        """Transform X onto PCA components"""
        
        X_transformed = np.dot(X - np.mean(X, axis=0), self.components_.T)
        return X_transformed
        
    def fit_transform(self, X: Union[np.ndarray, List[List]]) -> np.ndarray:
        """Fit to data and transform it"""
        
        self.fit(X)
        return self.transform(X)
