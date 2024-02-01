from memory_profiler import profile
import numpy as np
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
    def __init__(self, n_components, method='eigen'):
        self.n_components = n_components
        self.method = method

    @profile 
    def fit(self, X):
        # Center the data
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        if self.method == 'eigen':
            # Compute covariance matrix
            covariance_matrix = np.cov(X.T)

            # Eigen decomposition
            eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

            # Sort eigenvalues and eigenvectors
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Store first n_components eigenvectors
            self.components = eigenvectors[:, :self.n_components]

        elif self.method == 'svd':
            # Singular value decomposition
            _, _, VT = np.linalg.svd(X)

            # Store first n_components eigenvectors
            self.components = VT.T[:, :self.n_components]

        else:
            raise ValueError("Invalid method. Choose either 'eigen' or 'svd'.")

    @profile
    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.components)

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
