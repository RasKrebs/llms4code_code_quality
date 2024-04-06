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
from typing import Union

def page_rank(adjacency_matrix: np.ndarray, d: float = 0.85, iterations: int = 100, tol: float = 1e-3) -> np.ndarray:
    """
    Computes the PageRank of nodes in a graph given by its adjacency matrix.

    Parameters:
        adjacency_matrix (np.ndarray): The adjacency matrix of a graph, where entry ij represents whether there exists an edge from node i to node j.
        d (float): Damping factor for PageRank.  Default is 0.85.
        iterations (int): Maximum number of iterations for the algorithm to converge. Default is 100.
        tol (float): Minimal difference between PageRanks in consecutive iterations to consider the computation converged. Default is 1e-3.

    Returns:
        np.ndarray: The PageRank of each node in the graph.
    """
    N = len(adjacency_matrix) # Number of nodes

    # Step 1: Initialization
    ranks = np.ones(N) / N # Initially, all nodes have the same rank (1/n)
    teleportation = np.logical_not(np.any(adjacency_matrix, axis=0)) # Teleports to all nodes for dangling nodes

    # Step 2: Iteratively update PageRank until convergence or max iterations is reached
    for _ in range(iterations):
        old_ranks = ranks.copy()

        # Update PageRanks based on the damping factor, link structure and teleportation
        ranks = d * adjacency_matrix @ ranks + (1 - d) / N
        ranks += (d * (teleportation @ ranks)) / np.sum(teleportation)

        # Check for convergence
        if np.linalg.norm(ranks - old_ranks, 1) < tol:
            break

    return ranks
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

