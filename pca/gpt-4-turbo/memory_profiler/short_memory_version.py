
# IMPORTED FROM EVALUATE SCRIPT
from memory_profiler import profile
import os
import psutil

# Get the current process ID
pid = os.getpid()

# Create a psutil Process object for the current process
process = psutil.Process(pid)

# Get the number of logical CPUs in the system
num_cores = psutil.cpu_count(logical=True)

# BELOW IMPORTS ARE FROM THE SOLUTION
        
import numpy as np


@profile
def mean_normalize(data):
    # Subtract the mean from each feature (column)
    mean = np.mean(data, axis=0)
    normalized_data = data - mean
    return normalized_data

@profile
def covariance_matrix(data):
    # Calculate the covariance matrix
    cov_matrix = np.cov(data, rowvar=False)
    return cov_matrix

@profile
def eigen_decomposition(matrix):
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    # Sort the eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[idx]
    sorted_eigenvectors = eigenvectors[:, idx]
    return sorted_eigenvalues, sorted_eigenvectors

@profile
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
