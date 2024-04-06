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
        """
        Initialize the PCA class.

        Args:
            n_components (Optional[int], optional): Number of principal components to use. Defaults to None, which uses all components.
        """
        self.n_components = n_components  # Store number of components
        self.X_mean = None  # Mean for each feature
        self.components = None  # Eigenvectors (principal components)
        self.explained_variance_ratio = None  # Explained variance ratio for each component

    def fit(self, X: Union[list, np.ndarray]):
        """
        Fit the data by computing eigenvectors (principal components) and explained variance ratio.

        Args:
            X (Union[list, np.ndarray]): Input data as a list or numpy array.
        """
        if isinstance(X, list):  # If input data is a list, convert it to numpy array
            X = np.asarray(X)

        self.X_mean = np.mean(X, axis=0)  # Compute mean for each feature along rows (axis=0)
        X -= self.X_mean  # Mean-center the data by subtracting means from each row

        U, Sigma, Vt = np.linalg.svd(X, full_matrices=False)  # Singular value decomposition using numpy.linalg.svd (SVD)
        explained_variance = (Sigma ** 2) / (X.shape[0] - 1)  # Compute variance along each component
        total_variance = np.sum(explained_variance)  # Total variance

        self.explained_variance_ratio = explained_variance / total_variance  # Explained variance ratio for each component
        self.components = Vt.T[:self.n_components, :]  # Store the top n_components components (principal components)

    def transform(self, X: Union[list, np.ndarray]) -> np.ndarray:
        """
        Transform data by projecting it onto principal components and return transformed data.

        Args:
            X (Union[list, np.ndarray]): Input data as a list or numpy array.

        Returns:
            np.ndarray: Transformed data after projection on principal components.
        """
        if isinstance(X, list):  # If input data is a list, convert it to numpy array
            X = np.asarray(X)

        X -= self.X_mean  # Mean-center the data by subtracting means from each row
        transformed_data = np.dot(X, self.components.T)  # Project onto principal components and return transformed data
        return transformed_data

    def fit_transform(self, X: Union[list, np.ndarray]) -> np.ndarray:
        """
        Fit the data by computing eigenvectors (principal components) and explained variance ratio, then transform the data by projecting it onto principal components and return transformed data.

        Args:
            X (Union[list, np.ndarray]): Input data as a list or numpy array.

        Returns:
            np.ndarray: Transformed data after projection on principal components.
        """
        self.fit(X)  # Fit the data first by computing eigenvectors (principal components) and explained variance ratio
        transformed_data = self.transform(X)  # Then transform the data using fitted principal components
        return transformed_data
def execute():
    # Set the random seed for reproducibility
    np.random.seed(42)
    
    # Generate random data: 100 samples with 5 features
    X = np.random.rand(100, 5)
    
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

