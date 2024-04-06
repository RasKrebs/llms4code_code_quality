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

class PCA:
    """
    Principal Component Analysis (PCA) is a technique used for dimensionality reduction
    while preserving as much of the data's variation as possible.

    Attributes:
        n_components (int): Number of components to keep.
        components_ (np.ndarray): Principal axes in feature space.
        explained_variance_ (np.ndarray): Amount of variance explained by each of the selected components.
        explained_variance_ratio_ (np.ndarray): Percentage of variance explained by each of the selected components.
        cum_explained_variance_ratio_ (np.ndarray): Cumulative sum of explained variance ratio.
    """

    def __init__(self, n_components: int = None):
        """
        Initializes the PCA instance.

        Args:
            n_components (int, optional): Number of components to keep. If not specified, all components are kept.
        """
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.cum_explained_variance_ratio_ = None

    def fit(self, X: np.ndarray) -> None:
        """
        Fit the model with X by computing the eigenvalues and eigenvectors of the covariance matrix of X.

        Args:
            X (np.ndarray): The data to fit.
        """
        # Ensure X is a numpy array
        X = np.asarray(X, dtype=float)
        
        # Centering the data (subtracting the mean)
        X -= np.mean(X, axis=0)
        
        # Calculating the covariance matrix
        cov_matrix = np.cov(X, rowvar=False)

        # Eigen decomposition
        eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)
        
        # Sorting eigenvalues and eigenvectors in decreasing order
        sorted_idx = np.argsort(eigen_values)[::-1]
        eigen_values = eigen_values[sorted_idx]
        eigen_vectors = eigen_vectors[:, sorted_idx]
        
        # Store the principal components
        if self.n_components is not None and self.n_components < X.shape[1]:
            self.components_ = eigen_vectors[:, :self.n_components]
            self.explained_variance_ = eigen_values[:self.n_components]
        else:
            self.components_ = eigen_vectors
            self.explained_variance_ = eigen_values

        total_variance = eigen_values.sum()
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        self.cum_explained_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply dimensionality reduction to X using the computed principal components.

        Args:
            X (np.ndarray): The data to transform.

        Returns:
            np.ndarray: The transformed data.
        """
        X = np.asarray(X, dtype=float)
        X -= np.mean(X, axis=0)  # Centering is necessary before projection

        return np.dot(X, self.components_)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the model with X and apply the dimensionality reduction on X.

        Args:
            X (np.ndarray): The data to fit and transform.

        Returns:
            np.ndarray: The transformed data.
        """
        self.fit(X)
        return self.transform(X)
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

