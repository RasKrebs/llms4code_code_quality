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


def mean_normalize(data):
    # Subtract the mean from each feature (column)
    mean = np.mean(data, axis=0)
    normalized_data = data - mean
    return normalized_data

def covariance_matrix(data):
    # Calculate the covariance matrix
    cov_matrix = np.cov(data, rowvar=False)
    return cov_matrix

def eigen_decomposition(matrix):
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    # Sort the eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[idx]
    sorted_eigenvectors = eigenvectors[:, idx]
    return sorted_eigenvalues, sorted_eigenvectors

def pca(data, num_components):
    # Mean-normalize the data
    normalized_data = mean_normalize(data)
    
    # Calculate the covariance matrix
    cov_matrix = covariance_matrix(normalized_data)
    
    # Obtain eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigen_decomposition(cov_matrix)
    
    # Select top 'num_components' eigenvectors
    principal_components = eigenvectors[:, :num_components]
    
    # Project the data onto these principal components
    projected_data = np.dot(normalized_data, principal_components)
    
    return projected_data, principal_components, eigenvalues[:num_components]
def execute():
    # Set the random seed for reproducibility
    np.random.seed(42)
    
    # Generate random data: 100 samples with 5 features
    X = np.random.rand(100, 5)
    
    # Initialize PCA with 2 components
    pc = pca(X, num_components=2)
    
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

