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
from typing import Union, List

class PCA:
    def __init__(self, n_components: int = None):
        self.n_components = n_components
        self.eigenvalues = None
        self.explained_variance_ratio = None
        self.cumulative_explained_variance = None
        self.mean = None
        self.components = None

    def fit(self, X: Union[List[float], np.ndarray]) -> 'PCA':

        # Convert input to numpy array if necessary
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)

        # Calculate mean of each feature
        self.mean = np.mean(X, axis=0)

        # Subtract the mean from each data point
        X -= self.mean

        # Compute covariance matrix
        covariance_matrix = (1/(X.shape[0] - 1)) * np.dot(X.T, X)

        # Calculate eigen values and eigen vectors
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # Sort eigenvalues in descending order and sort corresponding eigen vectors accordingly
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Keep only the first n components
        if self.n_components is None:
            self.n_components = X.shape[1]

        self.explained_variance_ratio = (self.eigenvalues[:self.n_components]) / np.sum(self.eigenvalues)

        # Calculate cumulative sum of explained variance ratio
        self.cumulative_explained_variance = np.cumsum(self.explained_variance_ratio)

        # Normalize eigenvectors for unit vectors and keep only the required components
        self.components = (eigenvectors[:, :self.n_components] /
                             np.sqrt((eigenvectors[:, :self.n_components]**2).sum(axis=0))).T

        return self

    def transform(self, X: Union[List[float], np.ndarray]) -> np.ndarray:

        # Convert input to numpy array if necessary
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)

        # Subtract the mean from each data point
        X -= self.mean

        # Project data onto components
        projected_data = np.dot(X, self.components.T)

        return projected_data

    def fit_transform(self, X: Union[List[float], np.ndarray]) -> np.ndarray:

        # Fit PCA on the input data
        self = self.fit(X)

        # Transform the input data
        transformed_data = self.transform(X)

        return transformed_data
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

