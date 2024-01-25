import numpy as np
from scipy.linalg import svd

class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X, method='svd'):
        # center the data
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_

        if method == 'eigen':
            # compute covariance matrix
            cov_matrix = np.cov(X.T)

            # get eigenvalues and eigenvectors of the covariance matrix
            eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)

            # sort eigenvalues and corresponding eigenvectors in descending order
            idx = eig_vals.argsort()[::-1]
            self.eigenvalues_ = eig_vals[idx][:self.n_components]
            self.components_ = eig_vecs[:, idx][:, :self.n_components]

        elif method == 'svd':
            # perform SVD on the centered data matrix
            U, sigma, VT = svd(X)
            self.eigenvalues_ = (sigma ** 2)[:self.n_components]
            self.components_ = VT[:self.n_components].T

    def transform(self, X):
        # project data to lower dimensions
        return np.dot(X - self.mean_, self.components_.T)