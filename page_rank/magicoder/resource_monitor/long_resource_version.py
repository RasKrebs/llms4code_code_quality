import psutil
import os
import threading
import multiprocessing

# Define a global variable to store the maximum resources usage
max_resources_usage = {"cpu": 0, "memory": 0}


import numpy as np
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
from typing import Union

def page_rank(adjacency_matrix: Union[np.ndarray, list], d=0.85, max_iterations=100, tol=1e-6):
    """Computes PageRank for nodes in a directed graph represented by a numpy adjacency matrix.

    Parameters:
        adjacency_matrix (Union[np.ndarray, list]): The directed graph as an adjacency matrix where each element is either 0 or 1 representing whether there is an edge from i to j.
        d (float): Damping factor, defaults to 0.85.
        max_iterations (int): Maximum number of iterations in the PageRank computation, default is 100.
        tol (float): Convergence tolerance for PageRank values, default is 1e-6.

    Returns:
        page_rank(np.ndarray) : The computed page rank scores as a numpy array.
    """

    # Checking the input parameters and converting to numpy arrays if they are not already
    try:
        adjacency_matrix = np.asfarray(adjacency_matrix)
    except Exception as e:
        print("Error in Adjacency Matrix, please check the inputs")

    # Checking that the matrix is a square matrix
    if len(adjacency_matrix.shape) != 2 or adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
        print("Adjacency Matrix must be square")
        return None

    # Checking that the matrix is a directed graph (only non-negative entries allowed)
    if np.any(adjacency_matrix < 0):
        print("Adjacency Matrix should only contain non-negative values for directed graphs")
        return None

    # Initializing page rank scores, where each node starts with the same score (1/n)
    n = adjacency_matrix.shape[0]
    p = np.ones(n)/float(n)

    # Calculating the out-links for each node
    out_degree = adjacency_matrix.sum(axis=1)
    out_degree[out_degree==0] = 1 # For nodes without any outgoing edges, we consider them having a self loop
    teleportation_vector = np.ones((n,1))/float(n)

    for _ in range(max_iterations):
        new_p = (1-d)/n + d * np.dot(adjacency_matrix.T / out_degree, p) + d * teleportation_vector

        if np.sum((new_p - p)**2) < tol:  # Checking for convergence
            return new_p

        p = new_p

    print("Maximum iterations reached without reaching the desired tolerance")
    return p
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
    output = execute(graph)


    # Stop the monitoring
    monitoring = False
    monitor_thread.join()

    print(max_resources_usage['cpu']), print(max_resources_usage['memory'] / (1024**2))

