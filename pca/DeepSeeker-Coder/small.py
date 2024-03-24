import numpy as np

class PCA:
    def __init__(self, n_components, method='eigen'):
        self.n_components = n_components
        self.method = method

    def fit(self, X):
        # Center the data
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        if self.method == 'eigen':
            # Compute covariance matrix
            covariance_matrix = np.cov(X.T)

            # Eigen decomposition
            eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

            # Sort eigenvalues and eigenvectors
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Store first n_components eigenvectors
            self.components = eigenvectors[:, :self.n_components]

        elif self.method == 'svd':
            # Singular value decomposition
            _, _, VT = np.linalg.svd(X)

            # Store first n_components eigenvectors
            self.components = VT.T[:, :self.n_components]

        else:
            raise ValueError("Invalid method. Choose either 'eigen' or 'svd'.")

    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.components)