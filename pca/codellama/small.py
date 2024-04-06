import numpy as np

class PCA():
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        # Calculate mean vector of each feature
        self.mean_vec = np.mean(X, axis=0)

        # Subtract mean vector from original data matrix (centering)
        centered_X = X - self.mean_vec

        # Compute the covariance matrix
        cov_matrix = np.cov(centered_X.T)

        # Perform eigenvalue decomposition on the covariance matrix
        eig_vals, eig_vecs = np.linalg.eig(cov_matrix)

        # Arrange eigenvectors in descending order according to their corresponding eigenvalues (largest eigenvalue first)
        idx = eig_vals.argsort()[::-1]  # Find the indices that sort the array in descending order, then reverse those indices
        self.eig_vals = eig_vals[idx][:self.n_components]  # Keep the top n_components eigenvalues
        self.eig_vecs = eig_vecs[:, idx][:, :self.n_components]  # Keep the top n_components eigenvectors

    def transform(self, X):
        centered_X = X - self.mean_vec
        return np.dot(centered_X, self.eig_vecs)  # Transform new data using principal components

    def get_components(self):
        return self.eig_vecs, self.eig_vals
