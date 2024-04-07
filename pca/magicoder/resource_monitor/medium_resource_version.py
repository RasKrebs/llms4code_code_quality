import psutil
import os
import threading
import multiprocessing

# Define a global variable to store the maximum resources usage
max_resources_usage = {"cpu": 0, "memory": 0}


import numpy as np
np.random.seed(42)


# Implement the resource monitor
def resource_monitor():
    """
    Monitors the CPU and memory usage of the current process, updating global max usage.
    """
    global max_resources_usage
    process = psutil.Process(os.getpid())

    while monitoring:
        cpu_usage = process.cpu_percent(interval=1) / multiprocessing.cpu_count()
        memory_usage = process.memory_info().rss
        max_resources_usage['cpu'] = max(max_resources_usage['cpu'], cpu_usage)
        max_resources_usage['memory'] = max(max_resources_usage['memory'], memory_usage)



import numpy as np
from typing import Optional
from scipy.linalg import eigh

class PrincipalComponentAnalysis:
    """Class for principal component analysis"""

    def __init__(self, n_components:Optional[int]=None):
        self.n_components = n_components

    def fit(self, X:np.ndarray) -> None:
        """Fit the PCA model with input data"""
        # Subtract mean of each feature in dataset to get a zero-mean dataset
        self.X = X - np.mean(X, axis=0)

    def transform(self, X:np.ndarray) -> np.ndarray:
        """Transform input data using fitted model"""
        # Compute covariance matrix of the centered data
        cov_mat = np.cov(self.X, rowvar=False)

        # Eigendecomposition of the covariance matrix to get eigenvectors and eigenvalues
        eigval, self.eigvec = eigh(cov_mat)

        # Sort eigenpairs by eigenvalue in descending order
        idx = np.argsort(eigval)[::-1]
        self.eigval = eigval[idx]
        self.eigvec = self.eigvec[:,idx]

        if self.n_components is not None:
            # Select the first n_components eigenvectors (principal components)
            self.eigenvectors = self.eigvec[0:self.n_components, :]

        else:
            self.eigenvectors = self.eigvec

        # Transform the data onto the new feature space
        return np.dot(X - np.mean(X), self.eigenvectors.T).T

    def fit_transform(self, X:np.ndarray) -> np.ndarray:
        """Fit and transform input data"""
        self.fit(X)
        return self.transform(X)
def execute():
    # Set the random seed for reproducibility
    np.random.seed(42)

    # Generate random data: 10000 samples with 1000 features
    X = np.random.rand(10000, 1000)

    # Initialize PCA with 2 components
    pca = PrincipalComponentAnalysis(n_components=2)

    # Fit PCA on the generated data
    pca.fit(X)

    # Transform the data using the fitted PCA
    X_pca = pca.transform(X)

# Execute the function to see the results



if __name__ == "__main__":
    # Start the resource monitoring in a separate thread
    global monitoring
    monitoring = True
    monitor_thread = threading.Thread(target=resource_monitor)
    monitor_thread.start()

    # Execute the Huffman coding process

    # Using the execute function
    output = execute()


    # Stop the monitoring
    monitoring = False
    monitor_thread.join()

    print(max_resources_usage['cpu']), print(max_resources_usage['memory'] / (1024**2))

