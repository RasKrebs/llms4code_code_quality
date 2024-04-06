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
from typing import List, Tuple

def page_rank(adjacency_matrix: np.ndarray, damping: float = 0.85, iterations: int = 100, tolerance: float = 1e-6) -> np.ndarray:
    """
    Compute the PageRank of the nodes in the adjacency matrix.

    Parameters:
    adjacency_matrix (np.ndarray): The adjacency matrix of the graph.
    damping (float): The damping factor, usually set to 0.85.
    iterations (int): The maximum number of iterations to run the algorithm.
    tolerance (float): The minimum difference for the algorithm to consider the PageRank values converged.

    Returns:
    np.ndarray: The PageRank values of the nodes in the adjacency matrix.
    """

    # Get the number of nodes
    num_nodes = adjacency_matrix.shape[0]

    # Initialize the PageRank values
    page_rank_values = np.ones(num_nodes) / num_nodes

    # Compute the out-degree of each node
    out_degrees = adjacency_matrix.sum(axis=1)

    # Handle dangling nodes by teleporting to all nodes
    teleportation_matrix = np.ones((num_nodes, num_nodes)) / num_nodes

    # Iterate until convergence or max iterations
    for _ in range(iterations):
        old_page_rank_values = page_rank_values.copy()

        # Update the PageRank values
        page_rank_values = damping * np.dot(adjacency_matrix, page_rank_values / out_degrees) + (1 - damping) * np.dot(teleportation_matrix, page_rank_values)

        # Check for convergence
        if np.linalg.norm(page_rank_values - old_page_rank_values, 1) < tolerance:
            break

    return page_rank_values
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

