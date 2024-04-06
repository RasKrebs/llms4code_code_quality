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
    A class to perform Principal Component Analysis using eigen value decomposition.

    Attributes:
    n_components : int, optional
        Number of components to use. Default is all.

    Methods:
    fit(X: np.ndarray) -> None:
        Fit the PCA model with the data.
    
    transform(X: np.ndarray) -> np.ndarray:
        Apply dimensionality reduction to X.
    
    fit_transform(X: np.ndarray) -> np.ndarray:
        Fit the PCA model with the data and apply dimensionality reduction to X.
    """

    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X: np.ndarray) -> None:
        """
        Fit the PCA model with the data.
        
        Parameters:
        X : np.ndarray
            Input data of shape (n_samples, n_features).
        """
        # Centering the data
        mean = np.mean(X, axis=0)
        X_centered = X - mean
        
        # Calculate the covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)
        
        # Perform eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Sort eigenvalues and eigenvectors in decreasing order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Select number of components
        if self.n_components is not None:
            self.components = eigenvectors[:, :self.n_components]
        else:
            self.components = eigenvectors

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply dimensionality reduction to X.
        
        Parameters:
        X : np.ndarray
            Input data of shape (n_samples, n_features).
        
        Returns:
        np.ndarray
            Transformed data of shape (n_samples, n_components).
        """
        X_transformed = np.dot(X, self.components)
        return X_transformed

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the PCA model with the data and apply dimensionality reduction to X.
        
        Parameters:
        X : np.ndarray
            Input data of shape (n_samples, n_features).
        
        Returns:
        np.ndarray
            Transformed data of shape (n_samples, n_components).
        """
        self.fit(X)
        X_transformed = self.transform(X)
        return X_transformed
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

