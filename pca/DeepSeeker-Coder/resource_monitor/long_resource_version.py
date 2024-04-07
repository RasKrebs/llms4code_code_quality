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
    """
    Principal Component Analysis (PCA)

    Parameters
    ----------
    n_components : int, optional (default=None)
        Number of components to keep.
        If n_components is not set all components are kept.

    Attributes
    ----------
    components_ : array, [n_components, n_features]
        Principal axes in feature space, representing the directions of
        maximum variance in the data.

    explained_variance_ : array, [n_components]
        The amount of variance explained by each of the selected components.

    explained_variance_ratio_ : array, [n_components]
        Percentage of variance explained by each of the selected components.

    cumulative_explained_variance_ratio_ : array, [n_components]
        Cumulative sum of explained variance ratio.
    """

    def __init__(self, n_components: Optional[int] = None):
        self.n_components = n_components

    def fit(self, X: Union[np.ndarray, list]) -> None:
        """
        Fit the model with X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        """
        X = np.array(X)
        n_samples, n_features = X.shape

        # Center data
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_

        # Compute covariance matrix
        cov_matrix = np.cov(X.T)

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Sort eigenvalues and corresponding eigenvectors
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Determine number of components
        if self.n_components is None:
            self.n_components = n_features

        # Store results
        self.components_ = eigenvectors[:, :self.n_components]
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = self.explained_variance_ / np.sum(eigenvalues)
        self.cumulative_explained_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)

    def transform(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """
        Apply dimensionality reduction on X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_transformed : array, shape (n_samples, n_components)
            Projection of X in the first principal components space.
        """
        X = np.array(X)
        X -= self.mean_
        return np.dot(X, self.components_)

    def fit_transform(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """
        Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_transformed : array, shape (n_samples, n_components)
            Projection of X in the first principal components space.
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

