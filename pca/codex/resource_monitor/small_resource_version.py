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

def PCA(X, num_components):
    # 1. Standardize the dataset
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # 2. Compute the covariance matrix
    covariance_matrix = np.cov(X.T)

    # 3. Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # 4. Sort the eigenvalues and corresponding eigenvectors
    sorted_components = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_components]
    sorted_eigenvectors = eigenvectors[:,sorted_components]

    # 5. Select the first k eigenvectors
    eigenvector_subset = sorted_eigenvectors[:,0:num_components]

    # 6. Transform the data
    X_reduced = np.dot(eigenvector_subset.transpose(),X.transpose()).transpose()

    return X_reduced
def execute():
    # Set the random seed for reproducibility
    np.random.seed(42)

    # Generate random data: 10000 samples with 1000 features
    X = np.random.rand(10000, 1000)


    # Initialize PCA with 2 components
    pca = PCA(X, num_components=2)

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

