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
from typing import Union, Optional

class PCA:
    def __init__(self, n_components: Optional[int] = None):
        self.n_components = n_components
        self.eigenvalues = []
        self.explained_variance_ratio = []
        self.cumulative_explained_variance_ratio = []

    def fit(self, X: Union[np.ndarray, list]) -> None:
        """Compute PCA components for the data"""

        # Convert input to numpy array if it's a list
        if isinstance(X, list):
            X = np.array(X)

        # Normalize data by subtracting mean and dividing by standard deviation
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        X = (X - self.mean_) / self.std_

        # Get the covariance matrix
        cov_matrix = np.cov(X.T)

        # Compute eigenvalues and eigenvectors
        self.eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Sort eigenvalues and eigenvectors in descending order of the eigenvalues
        idx = self.eigenvalues.argsort()[::-1]
        self.eigenvalues = self.eigenvalues[idx]
        self.eigenvectors = eigenvectors[:,idx]

        # Calculate explained variance ratio and cumulative explained variance ratio
        self.explained_variance_ratio = (self.eigenvalues/sum(self.eigenvalues))*100
        self.cumulative_explained_variance_ratio = np.cumsum(self.explained_variance_ratio)

    def transform(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """Transform the data"""

        # Convert input to numpy array if it's a list
        if isinstance(X, list):
            X = np.array(X)

        # Normalize data by subtracting mean and dividing by standard deviation
        X = (X - self.mean_) / self.std_

        # Project onto the first n_components eigenvectors
        if self.n_components is not None:
            X = X @ self.eigenvectors[:, :self.n_components]
        else:
            X = X @ self.eigenvectors

        return X

    def fit_transform(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """Fit and transform the data"""

        self.fit(X)
        return self.transform(X)
def execute():
    # Set the random seed for reproducibility
    np.random.seed(42)

    # Generate random data: 10000 samples with 1000 features
    X = np.random.rand(10000, 1000)

    # Initialize PCA with 2 components
    pca = PCA(n_components=2)

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

