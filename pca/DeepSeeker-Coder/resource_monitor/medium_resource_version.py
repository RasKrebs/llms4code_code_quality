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

class PrincipalComponentAnalysis:
    """
    A class for performing Principal Component Analysis (PCA) on a dataset.

    Attributes:
    -----------
    n_components : int
        Number of principal components to keep.
    components : np.ndarray
        Principal components in feature space, representing the directions of maximum variance in the data.
    explained_variance : np.ndarray
        The amount of variance explained by each of the selected components.
    mean : np.ndarray
        Per-feature empirical mean, estimated from the training set.

    Methods:
    --------
    fit(X: np.ndarray) -> None:
        Computes the principal components and the explained variance.
    transform(X: np.ndarray) -> np.ndarray:
        Transforms the data using the computed principal components.
    fit_transform(X: np.ndarray) -> np.ndarray:
        Fits the model with X and applies the dimensionality reduction on X.
    """

    def __init__(self, n_components: Union[int, None] = None):
        """
        Parameters:
        -----------
        n_components : int, optional
            Number of components to keep. If n_components is not set all components are kept.
        """
        self.n_components = n_components
        self.components = None
        self.explained_variance = None
        self.mean = None

    def fit(self, X: np.ndarray) -> None:
        """
        Fit the model with X.

        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and n_features is the number of features.
        """
        # Center data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Compute covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort eigenvalues and corresponding eigenvectors
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Store explained variance
        self.explained_variance = eigenvalues

        # Store components
        self.components = eigenvectors.T

        # Reduce dimensions if n_components is set
        if self.n_components is not None:
            self.components = self.components[:self.n_components]
            self.explained_variance = self.explained_variance[:self.n_components]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply dimensionality reduction to X.

        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Data to transform.

        Returns:
        --------
        X_transformed : np.ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        # Center data
        X_centered = X - self.mean

        # Apply transformation
        X_transformed = np.dot(X_centered, self.components.T)

        return X_transformed

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the model with X and apply the dimensionality reduction on X.

        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and n_features is the number of features.

        Returns:
        --------
        X_transformed : np.ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        self.fit(X)
        return self.transform(X)
def execute():
    # Set the random seed for reproducibility
    np.random.seed(42)
    
    # Generate random data: 100 samples with 5 features
    X = np.random.rand(100, 5)
    
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

