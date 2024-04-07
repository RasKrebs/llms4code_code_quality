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
from typing import Optional, Tuple

def page_rank(graph: np.ndarray, d: float = 0.85, max_iterations: int = 100, tol: float = 1e-6) -> Tuple[np.ndarray, int]:
    """
    Computes PageRank for nodes in a directed graph represented by an adjacency matrix.

    Parameters:
        graph (numpy array): A square numpy array where entry [i, j] is the number of links from node i to node j.

        d (float, optional): Damping factor. Defaults to 0.85.

        max_iterations (int, optional): Maximum number of iterations. Defaults to 100.

        tol (float, optional): Tolerance for convergence. Defaults to 1e-6.

    Returns:
        Tuple[np.ndarray, int]: A tuple containing the PageRank vector and the number of iterations performed.
    """

    # Ensure input is a square numpy array
    assert graph.shape[0] == graph.shape[1], "Adjacency matrix should be square"

    # Initialize ranks to 1/n for all nodes
    n = graph.shape[0]
    ranks = np.ones(n) / n

    out_links = graph.sum(axis=1).reshape(-1, 1)
    dangling_nodes = (out_links == 0).flatten()

    # Handle dangling nodes and normalize the adjacency matrix

    M = np.divide(graph, out_links, out=np.zeros_like(graph, dtype=float), where=(out_links != 0))
    M[:, dangling_nodes] = 1 / n

    # Apply damping factor and teleportation
    M *= d
    M += (1 - d) / n

    for i in range(max_iterations):
        old_ranks = ranks.copy()

        # Iterate PageRank equation
        ranks = np.dot(M, ranks)

        if np.linalg.norm(ranks - old_ranks) < tol:
            break

    return ranks, i+1
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

