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

class PCA():
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        # Calculate mean vector of each feature
        self.mean_vec = np.mean(X, axis=0)

        # Subtract mean vector from original data matrix (centering)
        centered_X = X - self.mean_vec

        # Compute the covariance matrix
        cov_matrix = np.cov(centered_X.T)

        # Perform eigenvalue decomposition on the covariance matrix
        eig_vals, eig_vecs = np.linalg.eig(cov_matrix)

        # Arrange eigenvectors in descending order according to their corresponding eigenvalues (largest eigenvalue first)
        idx = eig_vals.argsort()[::-1]  # Find the indices that sort the array in descending order, then reverse those indices
        self.eig_vals = eig_vals[idx][:self.n_components]  # Keep the top n_components eigenvalues
        self.eig_vecs = eig_vecs[:, idx][:, :self.n_components]  # Keep the top n_components eigenvectors

    def transform(self, X):
        centered_X = X - self.mean_vec
        return np.dot(centered_X, self.eig_vecs)  # Transform new data using principal components

    def get_components(self):
        return self.eig_vecs, self.eig_vals
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

