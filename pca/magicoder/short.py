import numpy as np

def pca(data):
    # Center data by subtracting the mean value from each feature
    centered_data = data - np.mean(data, axis=0)

    # Compute covariance matrix
    cov_matrix = np.cov(centered_data.T)

    # Eigen decomposition of covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort the eigen values and corresponding eigen vectors in descending order
    indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:,indices]

    return sorted_eigenvectors 