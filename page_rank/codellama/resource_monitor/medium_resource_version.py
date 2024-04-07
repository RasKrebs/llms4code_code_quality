import psutil
import os
import threading
import multiprocessing

# Define a global variable to store the maximum resources usage
max_resources_usage = {"cpu": 0, "memory": 0}


import numpy as np
np.random.seed(42)

n = 100
graph = np.random.randint(0, 1, (n, n))
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



def page_rank(adjacency_matrix : np.ndarray, damping : float = 0.85, max_iterations : int = 100, tolerance : float = 1e-6) -> np.ndarray:
    """
    Calculates the PageRank of each node in a graph based on its adjacency matrix.

    Parameters
    ----------
    adjacency_matrix : np.ndarray
        The adjacency matrix representing the link structure between nodes. A value of 1 indicates a link, and 0 represents no link.
    damping : float, optional
        The damping factor for PageRank calculation (typically between 0.85 and 0.9). Defaults to 0.85.
    max_iterations : int, optional
        The maximum number of iterations to run before terminating if convergence is not reached. Defaults to 100.
    tolerance : float, optional
        A threshold for determining when the algorithm has converged. PageRank values should change by less than this amount between iterations. Defaults to 1e-6.

    Returns
    -------
    np.ndarray
        The PageRank value of each node in the graph, as an array with shape (n_nodes,).
    """

    # Validate input parameters
    if not isinstance(adjacency_matrix, np.ndarray):
        raise TypeError("Adjacency matrix must be a np ndarray.")
    if adjacency_matrix.dtype != bool:
        adjacency_matrix = adjacency_matrix.astype(bool)
        #raise ValueError("Adjacency matrix values must be boolean (1 represents a link, and 0 represents no link).")

    n_nodes = len(adjacency_matrix)
    rank_array = np.full((n_nodes,), 1 / n_nodes)  # Initialize ranks evenly

    for iteration in range(max_iterations):
        new_rank_array = (1 - damping) / n_nodes + damping * adjacency_matrix.T @ rank_array

        if np.linalg.norm(new_rank_array - rank_array) < tolerance:  # Convergence reached
            break

        rank_array = new_rank_array

    return rank_array

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

