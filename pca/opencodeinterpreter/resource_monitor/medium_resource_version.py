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
from typing import Union


class PrincipalComponentAnalysis:
    def __init__(self, n_components: Union[int, None] = None):
        self.n_components = n_components

    def fit(self, X: np.ndarray) -> 'PrincipalComponentAnalysis':
        # calculate mean of each feature
        mean = np.mean(X, axis=0)

        # subtract mean from the dataset
        self.X = X - mean

        # compute covariance matrix
        cov_mat = np.cov(self.X, rowvar=False)

        # compute eigen values and eigen vectors
        eig_values, eig_vectors = np.linalg.eigh(cov_mat)

        # sort eigen values in descending order and correspondingly
        # sort the eigen vectors
        idx = np.argsort(eig_values)[::-1]
        self.eigen_values = eig_values[idx]
        self.components = eig_vectors[:,idx]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # if n_components is not specified then use all the components
        n_features = X.shape[1]
        if self.n_components and self.n_components <= n_features:
            n_features = self.n_components

        return np.dot(X, self.components[:, :n_features])

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        # fit to the data and then transform it
        return self.fit(X).transform(X)

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

    return X_pca

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

