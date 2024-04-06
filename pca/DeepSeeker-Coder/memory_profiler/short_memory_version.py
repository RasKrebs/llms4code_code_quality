
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
def pca(X, num_components):
    # Step 1: Standardize the data
    X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # Step 2: Compute the covariance matrix
    cov_mat = np.cov(X_std.T)

    # Step 3: Compute the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov_mat)

    # Step 4: Sort the eigenvalues and their corresponding eigenvectors
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Step 5: Select the first num_components eigenvectors
    principal_components = eigenvectors[:, :num_components]

    # Step 6: Transform the original data
    X_transformed = np.dot(X_std, principal_components)

    return X_transformed
