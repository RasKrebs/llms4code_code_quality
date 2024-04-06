import psutil
import os
import threading
import multiprocessing

# Define a global variable to store the maximum resources usage
max_resources_usage = {"cpu": 0, "memory": 0}


np.random.seed(42)

n = 100
graph = np.random.randint(0, 2, (n, n))
# Ensure at least one outbound link per node by setting at least one 1 per row
for i in range(n):
    if not np.any(graph[i]):
        j = np.random.randint(0, n)
        graph[i][j] = 1


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
from typing import Optional, List

def page_rank(matrix: np.ndarray, damping: float = 0.85, iterations: int = 100, tolerance: float = 0.0001) -> np.ndarray:
    """
    Calculate PageRank from an adjacency matrix with a specified number of iterations and a tolerance level for the convergence.

    Parameters:
        matrix (np.ndarray): A numpy array representing the adjacency matrix of the webgraph.
        damping (float): The probability that user will follow one outbound link in the matrix, default is 0.85.
        iterations (int): The number of iterations to run PageRank algorithm, default is 100.
        tolerance (float): The minimal change in pageranks required for a new iteration to start, default is 0.0001.

    Returns:
        np.ndarray: A numpy array representing the calculated pagerank vector with the same order as the nodes in the graph.
    """
    # Check if matrix is valid adjacency matrix
    assert(matrix.shape[0] == matrix.shape[1])

    # Number of Nodes
    N = len(matrix)

    # Initializing PageRank Vector
    pr_vector = np.full((N, 1), fill_value=1/N).astype('float64')

    for _ in range(iterations):
        new_pr_vector = (1 - damping)/N + damping * np.dot(matrix, pr_vector)

        if np.sum(np.abs(new_pr_vector - pr_vector)) <= tolerance:
            return new_pr_vector

        pr_vector = new_pr_vector

    # If max iterations reached without convergence
    return pr_vector
def execute(adj_matrix):
    # Execute the page_rank function
    output = page_rank(adj_matrix)
    return output



if __name__ == "__main__":
    # Start the resource monitoring in a separate thread
    global monitoring
    monitoring = True
    monitor_thread = threading.Thread(target=resource_monitor)
    monitor_thread.start()

    # Execute the Huffman coding process

    # Using the execute function
    output = execute(adj_matrix)


    # Stop the monitoring
    monitoring = False
    monitor_thread.join()

    print(max_resources_usage['cpu']), print(max_resources_usage['memory'] / (1024**2))

