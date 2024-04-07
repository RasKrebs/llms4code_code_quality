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

def PCA(X, k):
  """
  Performs PCA on data matrix X and returns projected data and explained variance.

  Args:
    X: Data matrix (n x m).
    k: Number of principal components to retain.

  Returns:
    X_reduced: Projected data matrix (n x k).
    explained_variance: Explained variance ratio for each component.
  """
  # Center data (optional)
  X_centered = X - np.mean(X, axis=0)

  # Covariance matrix
  covariance = np.cov(X_centered.T)

  # Eigenvalue decomposition
  eigenvalues, eigenvectors = np.linalg.eig(covariance)

  # Sort eigenvectors by eigenvalues (descending)
  sorted_index = np.argsort(eigenvalues)[::-1]
  eigenvalues = eigenvalues[sorted_index]
  eigenvectors = eigenvectors[:, sorted_index]

  # Select top k eigenvectors
  V_reduced = eigenvectors[:, :k]

  # Project data onto principal components
  X_reduced = X_centered @ V_reduced

  # Explained variance ratio
  explained_variance = eigenvalues / np.sum(eigenvalues)

  return X_reduced, explained_variance

def execute():
    # Set the random seed for reproducibility
    np.random.seed(42)

    # Generate random data: 10000 samples with 1000 features
    X = np.random.rand(10000, 1000)


    # Initialize PCA with 2 components
    pca = PCA(X, 2)

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

