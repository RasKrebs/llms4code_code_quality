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
    Principal Component Analysis (PCA) class for dimensionality reduction.

    This class performs PCA on a dataset using eigenvalue decomposition.

    Attributes:
        n_components (int): The number of principal components to use. Defaults to all.
        components_ (np.ndarray): The eigenvectors representing the principal components.
        explained_variance_ratio_ (np.ndarray): The explained variance ratio for each component.
        explained_variance_ (np.ndarray): The explained variance for each component.
        cumulative_explained_variance_ratio_ (np.ndarray): 
            The cumulative explained variance ratio for each component.
    """

    def __init__(self, n_components: int = None) -> None:
        """
        Initializes the PCA object.

        Args:
            n_components (int, optional): The number of principal components to use. 
                Defaults to all components.
        """
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ratio_ = None
        self.explained_variance_ = None
        self.cumulative_explained_variance_ratio_ = None

    def fit(self, X: np.ndarray) -> None:
        """
        Fits the PCA model to the data X.

        This method computes the eigen values, eigen vectors, explained variance ratio,
        and cumulative explained variance ratio for the data.

        Args:
            X (np.ndarray): The input data of shape (n_samples, n_features).
        """

        # Handle different data types by converting to float
        X = X.astype(np.float64)

        # Center the data by subtracting the mean
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Compute the covariance matrix
        covariance = np.cov(X_centered.T)

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(covariance)

        # Sort eigenvalues and eigenvectors in descending order
        sorted_index = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_index]
        eigenvectors = eigenvectors[:, sorted_index]

        # Select the desired number of components
        if self.n_components is None:
            self.n_components = len(eigenvalues)
        components = eigenvectors[:, : self.n_components]

        # Calculate explained variance ratio and cumulative sum
        total_variance = np.sum(eigenvalues)
        explained_variance = eigenvalues[: self.n_components]
        explained_variance_ratio = explained_variance / total_variance
        cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)

        # Store fit results
        self.components_ = components
        self.explained_variance_ratio_ = explained_variance_ratio
        self.explained_variance_ = explained_variance
        self.cumulative_explained_variance_ratio_ = cumulative_explained_variance_ratio

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transforms the data X onto the principal components.

        Args:
            X (np.ndarray): The input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: The transformed data of shape (n_samples, n_components).
        """

        # Check if PCA is fitted
        if self.components_ is None:
            raise ValueError("PCA object is not fitted. Call fit(X) before transform.")

        # Handle different data types by converting to float
        X = X.astype(np.float64)

        # Center the data using the mean from fit
        mean = self.mean  # Use stored mean if available
        if mean is None:
            raise ValueError(
                "PCA object doesn't have mean information. Call fit(X) with the same data used for fitting."
            )
        X_centered = X - mean

        # Project data onto components
        return X_centered @ self.components_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
            """
            Fits the PCA model to the data X and then transforms it.

            This method is equivalent to calling fit(X) followed by transform(X).

            Args:
                X (np.ndarray): The input data of shape (n_samples, n_features).

            Returns:
                np.ndarray: The transformed data of shape (n_samples, n_components).
            """

            # Call fit method to compute necessary components
            self.fit(X)

            # Call transform method to project data onto components
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

