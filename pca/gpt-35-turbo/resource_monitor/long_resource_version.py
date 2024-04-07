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



class PCA:
    def __init__(self, n_components=None):
        """
        Initialize PCA object with number of components.

        Parameters:
        n_components (int): Number of components to use. Default is None.
        """
        self.n_components = n_components
        self.components = None
        self.eigenvalues = None
        self.explained_variance_ratio = None
        self.cumulative_explained_variance_ratio = None

    def fit(self, X):
        """
        Fit the data and compute eigenvalues, eigenvectors, explained variance ratio and cumulative explained variance ratio.

        Parameters:
        X (numpy array): Input data

        Returns:
        None
        """
        # Ensure X is a numpy array
        if type(X) != np.ndarray:
            X = np.array(X)

        # Center the data
        X_mean = np.mean(X, axis=0)
        X_centered = X - X_mean

        # Compute covariance matrix
        cov_mat = np.cov(X_centered.T)

        # Compute eigenvalues and eigenvectors
        self.eigenvalues, self.components = np.linalg.eig(cov_mat)

        # Sort components based on eigenvalues
        idx = np.argsort(self.eigenvalues)[::-1]
        self.components = self.components[:, idx]
        self.eigenvalues = self.eigenvalues[idx]

        # Compute explained variance ratio
        total_variance = np.sum(self.eigenvalues)
        self.explained_variance_ratio = self.eigenvalues / total_variance

        # Compute cumulative explained variance ratio
        self.cumulative_explained_variance_ratio = np.cumsum(self.explained_variance_ratio)

    def transform(self, X):
        """
        Project input data onto the components.

        Parameters:
        X (numpy array): Input data

        Returns:
        X_projected (numpy array): Projected data
        """
        # Ensure X is a numpy array
        if type(X) != np.ndarray:
            X = np.array(X)

        # Center the data
        X_mean = np.mean(X, axis=0)
        X_centered = X - X_mean

        # Project data onto components
        X_projected = np.dot(X_centered, self.components)

        return X_projected

    def fit_transform(self, X):
        """
        Fit the data and transform it using the computed components.

        Parameters:
        X (numpy array): Input data

        Returns:
        X_projected (numpy array): Projected data
        """
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

