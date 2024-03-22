import numpy as np
from memory_profiler import profile
import psutil
import os

# Get the current process ID
pid = os.getpid()

# Create a psutil Process object for the current process
process = psutil.Process(pid)

# Get the number of logical CPUs in the system
num_cores = psutil.cpu_count(logical=True)

@profile
def standardize_data(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

@profile
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

@profile
def pca_svd(X, num_components):
    # Standardize the data
    X_std = standardize_data(X)
    # Compute SVD
    U, S, Vt = np.linalg.svd(X_std)
    # Select top num_components eigenvectors (right singular vectors)
    V = Vt.T
    return X_std.dot(V[:, :num_components])

# Execute script data to get memory usage
if __name__ == "__main__":
    # For reproducibility
    np.random.seed(42)

    # Generate random data
    data = np.random.rand(10000, 100)
    
    # Create PCA object
    pca = pca_eigen(data, 10)
    pca = pca_svd(data, 10)

    
    # Get the CPU percentage usage of the process
    cpu_usage = process.cpu_percent(interval=1)/ num_cores
    print(f"CPU Usage: {cpu_usage}%")

