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
from scipy import linalg

def pca(X, k):
    # Step 1: Calculate the mean value for each feature
    means = X.mean(axis=0)

    # Step 2: Compute the covariance matrix
    cov_matrix = np.cov((X - means), rowvar=False)

    # Step 3: Compute the eigenvalues and eigenvectors for this covariance matrix
    eig_values, eig_vectors = linalg.eigh(cov_matrix)

    # Sort the eigenvalues in descending order along with the corresponding eigenvectors
    sorted_indexes = np.argsort(eig_values)[::-1]
    sorted_eigenvalues = eig_values[sorted_indexes]
    sorted_eigenvectors = eig_vectors[:, sorted_indexes]

    # Select the top k eigenvectors for transformation
    top_k_vectors = sorted_eigenvectors[:, :k]

    return means, top_k_vectors
def execute():
    # Set the random seed for reproducibility
    np.random.seed(42)
    
    # Generate random data: 100 samples with 5 features
    X = np.random.rand(100, 5)
    
    # Initialize PCA with 2 components
    princ = pca(X, 2)
    
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

