from memory_profiler import profile
import numpy as np
from scipy.linalg import svd
import psutil
import os

# Get the current process ID
pid = os.getpid()

# Create a psutil Process object for the current process
process = psutil.Process(pid)

# Get the number of logical CPUs in the system
num_cores = psutil.cpu_count(logical=True)

class PCA:
    @profile
    def __init__(self, n_components=None):
        self.n_components = n_components
    @profile
    def fit(self, X, method='eigen'):
        # center the data
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_

        if method == 'eigen':
            # compute covariance matrix
            cov_matrix = np.cov(X.T)

            # get eigenvalues and eigenvectors of the covariance matrix
            eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)

            # sort eigenvalues and corresponding eigenvectors in descending order
            idx = eig_vals.argsort()[::-1]
            self.eigenvalues_ = eig_vals[idx][:self.n_components]
            self.components_ = eig_vecs[:, idx][:, :self.n_components]

        elif method == 'svd':
            # perform SVD on the centered data matrix
            U, sigma, VT = svd(X)
            self.eigenvalues_ = (sigma ** 2)[:self.n_components]
            self.components_ = VT[:self.n_components].T
    @profile
    def transform(self, X):
        # project data to lower dimensions
        return np.dot(X - self.mean_, self.components_)

# Execute script data to get memory usage
if __name__ == "__main__":
    # For reproducibility
    np.random.seed(42)

    # Generate random data
    data = np.random.rand(10000, 100)
    
    # Create PCA object
    pca = PCA(n_components=10)
    pca.fit(data)
    pca = pca.transform(data)
    
    # Get the CPU percentage usage of the process
    cpu_usage = process.cpu_percent(interval=1)/ num_cores
    print(f"CPU Usage: {cpu_usage}%")

