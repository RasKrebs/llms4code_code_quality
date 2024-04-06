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



from typing import Optional, Tuple

import numpy as np


class PrincipalComponentAnalysis:
    """
    A class implementing Principal Component Analysis (PCA) from scratch using eigen-decomposition.

    This class performs dimensionality reduction on data by finding the principal components,
    which capture the most significant variance in the data.

    Attributes:
        eigenvectors_: (np.ndarray): The eigenvectors of the covariance matrix.
        explained_variance_: (np.ndarray): The explained variance ratio for each principal component.
        n_components_: (int): The number of principal components used.
    """

    def __init__(self, n_components: Optional[int] = None) -> None:
        """
        Initializes the PCA object.

        Args:
            n_components: (Optional[int]): The number of principal components to use. Defaults to all.
        """
        self.eigenvectors_: np.ndarray = None
        self.explained_variance_: np.ndarray = None
        self.n_components_: int = n_components

    def fit(self, X: np.ndarray) -> None:
        """
        Fits the PCA model to the data.

        This method calculates the covariance matrix, performs eigen-decomposition, and stores
        the eigenvectors and explained variance ratio.

        Args:
            X: (np.ndarray): The data matrix with shape (n_samples, n_features).
        """
        n_samples, n_features = X.shape

        # Center the data
        X_centered = X - np.mean(X, axis=0)

        # Calculate the covariance matrix
        covariance = np.cov(X_centered.T)

        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eig(covariance)

        # Sort eigenvalues and eigenvectors by decreasing order of eigenvalues
        sorted_index = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_index]
        eigenvectors = eigenvectors[:, sorted_index]

        # Determine the number of components to use
        if self.n_components_ is None:
            self.n_components_ = n_features
        else:
            self.n_components_ = min(self.n_components_, n_features)

        # Select the top n_components eigenvectors and explained variance
        self.eigenvectors_ = eigenvectors[:, : self.n_components_]
        self.explained_variance_ = eigenvalues[: self.n_components_] / np.sum(eigenvalues)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transforms the data using the fitted PCA model.

        This method projects the data onto the principal components.

        Args:
            X: (np.ndarray): The data matrix with shape (n_samples, n_features).

        Returns:
            (np.ndarray): The transformed data matrix with shape (n_samples, n_components).
        """
        if self.eigenvectors_ is None:
            raise ValueError("PCA model not fitted. Call fit before transform.")

        # Center the data
        X_centered = X - np.mean(X, axis=0)

        # Project data onto principal components
        return X_centered @ self.eigenvectors_

    def fit_transform(self, X: np.ndarray) -> Tuple[np.ndarray, None]:
        """
        Fits the PCA model to the data and then transforms it.

        This method is a convenience function that combines both fit and transform steps.

        Args:
            X: (np.ndarray): The data matrix with shape (n_samples, n_features).

        Returns:
            Tuple[np.ndarray, None]: The transformed data matrix with shape (n_samples, n_components)
                                      and None (for compatibility with scikit-learn).
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

