
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
