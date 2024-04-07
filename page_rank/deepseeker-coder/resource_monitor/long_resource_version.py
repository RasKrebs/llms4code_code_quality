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

def page_rank(graph: np.ndarray, d: float = 0.85, max_iterations: int = 100, tol: float = 1e-6) -> Union[np.ndarray, None]:
    """
    Compute PageRank for nodes in a directed graph represented by a numpy adjacency matrix.

    Parameters:
    graph (np.ndarray): A square numpy array representing the adjacency matrix of the graph.
    d (float): Damping factor, default is 0.85.
    max_iterations (int): Maximum number of iterations, default is 100.
    tol (float): Convergence tolerance, default is 1e-6.

    Returns:
    np.ndarray: A numpy array representing the PageRank of the nodes in the graph.
    None: If the input graph is not a square numpy array or if the graph contains negative values.
    """

    # Check if the input graph is a square numpy array
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        print("Error: The input graph must be a square numpy array.")
        return None

    # Check if the graph contains negative values
    if np.any(graph < 0):
        print("Error: The graph must not contain negative values.")
        return None

    # Initialize the ranks
    ranks = np.ones(graph.shape[0]) / graph.shape[0]

    # Iterate until convergence or reaching the max iterations
    for _ in range(max_iterations):
        old_ranks = ranks.copy()

        # Adjust for outbound links and the damping factor
        ranks = d * np.dot(graph, ranks)

        # Factor in teleportation for nodes without outbound links
        ranks += (1 - d) / graph.shape[0]

        # Check for convergence
        if np.linalg.norm(ranks - old_ranks) < tol:
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
    output = execute(graph)


    # Stop the monitoring
    monitoring = False
    monitor_thread.join()

    print(max_resources_usage['cpu']), print(max_resources_usage['memory'] / (1024**2))

