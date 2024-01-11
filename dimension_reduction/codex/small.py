import numpy as np

class PCA:
    def __init__(self, n_components, method='eigen'):
        self.n_components = n_components
        self.method = method
        self.components = None
        self.mean = None

    def fit(self, X):
        # Mean centering
        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        # covariance, function needs samples as columns
        cov = np.cov(X.T)

        # eigen decomposition
        if self.method == 'eigen':
            eigenvalues, eigenvectors = np.linalg.eig(cov)
            # -> eigenvector v = [:,i] column vector, transpose for easier calculations
            # sort eigenvectors
            eigenvectors = eigenvectors.T
            idxs = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idxs]
            eigenvectors = eigenvectors[idxs]
            # store first n eigenvectors
            self.components = eigenvectors[0:self.n_components]

        # singular value decomposition
        elif self.method == 'svd':
            u, s, v = np.linalg.svd(X.T)
            self.components = v[0:self.n_components]

    def transform(self, X):
        # project data
        X = X - self.mean
        return np.dot(X, self.components.T)