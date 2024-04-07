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
from typing import Union, Tuple

class PCA:
    """Principal Component Analysis"""

    def __init__(self, n_components: int=None):
        self.n_components = n_components  # number of components to use

    def fit(self, X: Union[np.ndarray, list]) -> None:
        """Fit the data."""
        if isinstance(X, list):
            X = np.array(X)

        self._mean = np.mean(X, axis=0)  # calculate mean for subtraction of mean value

        X_std = X - self._mean  # subtract mean from each dimension to center the data

        cov_mat = np.cov(X_std, rowvar=False)
        eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)

        sorted_indexes = np.argsort(eigen_values)[::-1]  # sort eigenvalues in decreasing order
        self._sorted_eigenvalue = eigen_values[sorted_indexes]
        self._sorted_eigenvectors = eigen_vectors[:, sorted_indexes]

        total = sum(self._sorted_eigenvalue)  # Total of all eigenvalues

        if self.n_components:
            self.explained_variance_ratio = [(i / total)*100 for i in self._sorted_eigenvalue[:self.n_components]]
        else:
            self.explained_variance_ratio = 100 * (self._sorted_eigenvalue/total)  # all eigenvalues

    def transform(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """Project the input data onto the components."""
        if isinstance(X, list):
            X = np.array(X)

        X_std = X - self._mean  # subtract mean from each dimension to center the data

        return np.dot(self._sorted_eigenvectors[:, :self.n_components].T, X_std.T).T

    def fit_transform(self, X: Union[np.ndarray, list]) -> Tuple[Union[float, list, np.ndarray], np.ndarray]:
        """Fit the data and project it."""
        self.fit(X)

        return (self._sorted_eigenvalue[:self.n_components],  # eigenvalues of first n components
                self.transform(X))

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

