import numpy as np
from modules.helpers import pca_data

def pca_eigen_decomp(X, num_components):
    # Center the data
    mean = np.mean(X, axis=0)
    centered_X = X - mean

    # Calculate the covariance matrix
    cov_matrix = np.cov(centered_X, rowvar=False)

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort eigenvalues and corresponding eigenvectors
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Select the top 'num_components' eigenvectors
    top_eigenvectors = eigenvectors[:, :num_components]

    # Project the data onto the selected eigenvectors
    pca_result = np.dot(centered_X, top_eigenvectors)

    return pca_result


def pca_svd(X, num_components):
    # Center the data
    mean = np.mean(X, axis=0)
    centered_X = X - mean

    # Perform SVD
    U, S, Vt = np.linalg.svd(centered_X, full_matrices=False)

    # Select the top 'num_components' singular values and corresponding columns of U
    U_reduced = U[:, :num_components]

    # Project the data onto the selected components
    pca_result = np.dot(centered_X, U_reduced)

    return pca_result

if __name__ == "__main__":
    # Set random seed for reproducibility
    X_random, X_bodyfat = pca_data()

    # Test the two methods
    pca_eigen_decomp(X_random, 2)
    pca_eigen_decomp(X_bodyfat, 2)
    pca_svd(X_random, 2)
    pca_svd(X_bodyfat, 2)