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



from typing import Union, Optional
import numpy as np

class PrincipalComponentAnalysis:
    """Principal Component Analysis implementation.

    Args:
        n_components (int): Number of components to use. Defaults to None.

    Attributes:
        X_transformed (np.ndarray): Transformed data after fit and transform.
        components (np.ndarray): Principal components of the dataset.

    """
    def __init__(self, n_components: Optional[int] = None) -> None:
        self.n_components = n_components
        self.X_transformed = None
        self.components = None

    def fit(self, X: np.ndarray):
        """Fits the PCA model to the data using eigen value decomposition.

        Args:
            X (np.ndarray): Input dataset with shape (n_samples, n_features).

        Returns:
            self: The fitted instance of PrincipalComponentAnalysis class.

        """
        # Calculate the mean of each feature column
        self._mean = np.mean(X, axis=0)
        # Center the data by subtracting the mean
        X_centered = X - self._mean

        if self.n_components:
            # Use n_components argument for eigen value decomposition
            eigenvalues, eigenvectors = np.linalg.eig(np.cov(X_centered.T))
        else:
            # Use all components for eigen value decomposition
            eigenvalues, eigenvectors = np.linalg.eig(np.cov(X_centered.T))

        # Sort the eigenvalues and eigenvectors in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        self.components = eigenvectors

    def transform(self, X: np.ndarray) -> Union[np.ndarray, None]:
        """Transforms the data using the fitted principal components.

        Args:
            X (np.ndarray): Input dataset with shape (n_samples, n_features).

        Returns:
            np.ndarray or None: Transformed data after applying PCA to input.

        """
        # Center the data by subtracting the mean
        X_centered = X - self._mean
        # Project the centered data onto principal components
        self.X_transformed = np.dot(X_centered, self.components[:, :self.n_components])

    def fit_transform(self, X: np.ndarray) -> Union[np.ndarray, None]:
        """Fits and transforms the data using eigen value decomposition.

        Args:
            X (np.ndarray): Input dataset with shape (n_samples, n_features).

        Returns:
            np.ndarray or None: Transformed data after applying PCA to input.

        """
        self.fit(X)
        self.transform(X)
        return self.X_transformed
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

