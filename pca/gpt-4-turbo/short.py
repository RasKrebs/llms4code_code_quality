import numpy as np


def mean_normalize(data):
    # Subtract the mean from each feature (column)
    mean = np.mean(data, axis=0)
    normalized_data = data - mean
    return normalized_data

def covariance_matrix(data):
    # Calculate the covariance matrix
    cov_matrix = np.cov(data, rowvar=False)
    return cov_matrix

def eigen_decomposition(matrix):
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    # Sort the eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[idx]
    sorted_eigenvectors = eigenvectors[:, idx]
    return sorted_eigenvalues, sorted_eigenvectors

def pca(data, num_components):
    # Mean-normalize the data
    normalized_data = mean_normalize(data)
    
    # Calculate the covariance matrix
    cov_matrix = covariance_matrix(normalized_data)
    
    # Obtain eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigen_decomposition(cov_matrix)
    
    # Select top 'num_components' eigenvectors
    principal_components = eigenvectors[:, :num_components]
    
    # Project the data onto these principal components
    projected_data = np.dot(normalized_data, principal_components)
    
    return projected_data, principal_components, eigenvalues[:num_components]
