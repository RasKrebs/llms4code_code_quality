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

class PrincipalComponentAnalysis:
    """
    Principal Component Analysis (PCA) class.
    PCA is a dimensionality reduction technique that is widely used in practice.
    This class implements PCA using eigenvalue decomposition.
    """

    def __init__(self, n_components: int = None):
        """
        Initialize the PCA object.

        :param n_components: The number of components to use. If not specified, all components are used.
        """
        self.n_components = n_components
        self.eigenvalues = None
        self.eigenvectors = None

    def fit(self, X: np.ndarray):
        """
        Fit the PCA model with the input data.

        :param X: The input data, a 2D numpy array of shape (n_samples, n_features).
        """
        # Center the data
        X_centered = X - np.mean(X, axis=0)

        # Compute the covariance matrix
        covariance_matrix = np.cov(X_centered.T)

        # Compute the eigenvalues and eigenvectors
        self.eigenvalues, self.eigenvectors = np.linalg.eig(covariance_matrix)

        # Sort the eigenvectors by decreasing eigenvalues
        idx = np.argsort(self.eigenvalues)[::-1]
        self.eigenvalues = self.eigenvalues[idx]
        self.eigenvectors = self.eigenvectors[:, idx]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply dimensionality reduction to the input data.

        :param X: The input data, a 2D numpy array of shape (n_samples, n_features).
        :return: The transformed data.
        """
        # Center the data
        X_centered = X - np.mean(X, axis=0)

        # Project the data onto the principal components
        if self.n_components is None:
            return np.dot(X_centered, self.eigenvectors)
        else:
            return np.dot(X_centered, self.eigenvectors[:, :self.n_components])

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the PCA model with the input data and apply dimensionality reduction.

        :param X: The input data, a 2D numpy array of shape (n_samples, n_features).
        :return: The transformed data.
        """
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

