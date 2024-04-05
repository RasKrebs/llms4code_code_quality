import numpy as np
from scipy import linalg

def pca(X, k):
    # Step 1: Calculate the mean value for each feature
    means = X.mean(axis=0)

    # Step 2: Compute the covariance matrix
    cov_matrix = np.cov((X - means), rowvar=False)

    # Step 3: Compute the eigenvalues and eigenvectors for this covariance matrix
    eig_values, eig_vectors = linalg.eigh(cov_matrix)

    # Sort the eigenvalues in descending order along with the corresponding eigenvectors
    sorted_indexes = np.argsort(eig_values)[::-1]
    sorted_eigenvalues = eig_values[sorted_indexes]
    sorted_eigenvectors = eig_vectors[:, sorted_indexes]

    # Select the top k eigenvectors for transformation
    top_k_vectors = sorted_eigenvectors[:, :k]

    return means, top_k_vectors