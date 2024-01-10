import numpy as np

def standardize_data(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

def pca_eigen(X, num_components):
    # Standardize the data
    X_std = standardize_data(X)
    # Covariance matrix
    covariance_matrix = np.cov(X_std.T)
    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    # Sort eigenvectors by eigenvalues in descending order
    idx = eigenvalues.argsort()[::-1]   
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    # Select top num_components eigenvectors
    eigenvectors = eigenvectors[:, :num_components]
    return X_std.dot(eigenvectors)

def pca_svd(X, num_components):
    # Standardize the data
    X_std = standardize_data(X)
    # Compute SVD
    U, S, Vt = np.linalg.svd(X_std)
    # Select top num_components eigenvectors (right singular vectors)
    V = Vt.T
    return X_std.dot(V[:, :num_components])

# Example usage
X = np.array([[1, 2], [3, 4], [5, 6]])  # Replace with your dataset
num_components = 1  # Number of components you want to keep

# Using Eigen Decomposition
pca_eigen_results = pca_eigen(X, num_components)

# Using SVD
pca_svd_results = pca_svd(X, num_components)
