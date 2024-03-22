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
def pca_eigen(X, k):
    # Standardize the data
    X = (X - np.mean(X)) / np.std(X)

    # Compute the covariance matrix
    cov = np.cov(X, rowvar=False)

    # Compute the eigenvalues and eigenvectors of the covariance matrix
    eigvals, eigvecs = np.linalg.eig(cov)

    # Sort the eigenvalues and corresponding eigenvectors in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Select the top k eigenvectors that explain the most variance
    W = eigvecs[:, :k]

    # Project the data onto the selected eigenvectors
    X_pca = X @ W

    return X_pca

@profile
def pca_svd(X, k):
    # Standardize the data
    X = (X - np.mean(X)) / np.std(X)

    # Compute the singular value decomposition of the data matrix
    U, S, V = np.linalg.svd(X, full_matrices=False)

    # Select the top k singular values and corresponding singular vectors
    S = S[:k]
    U = U[:, :k]
    V = V[:, :k]

    # Compute the principal components
    X_pca = U @ np.diag(S) @ V.T

    return X_pca


# Execute script data to get memory usage
if __name__ == "__main__":
    # For reproducibility
    np.random.seed(42)

    # Generate random data
    data = np.random.rand(10000, 100)
    
    # Create PCA object
    pca = pca_eigen(X=data, k=10)

    # Get the CPU percentage usage of the process
    cpu_usage = process.cpu_percent(interval=1)/ num_cores
    print(f"CPU Usage: {cpu_usage}%")
