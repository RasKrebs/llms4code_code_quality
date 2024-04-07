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

def compute_PageRank(graph: np.ndarray, d: float = 0.85, max_iterations: int = 100, tol: float = 1e-6) -> np.ndarray:
    """
    Compute PageRank for nodes in a directed graph represented by a numpy adjacency matrix.

    Parameters:
    graph: numpy adjacency matrix representing the graph.
    d: damping factor (default 0.85).
    max_iterations: maximum number of iterations (default 100).
    tol: convergence tolerance (default 1e-6).

    Returns:
    PageRank as a numpy array.
    """
    # Check input parameters
    if not 0 < d < 1:
        raise ValueError("Damping factor must be between 0 and 1.")
    if not max_iterations > 0:
        raise ValueError("Max iterations must be positive.")
    if not tol > 0:
        raise ValueError("Tolerance must be positive.")

    # Number of nodes
    N = graph.shape[0]

    # Initialize ranks
    ranks = np.full(N, 1/N)

    # Dangling nodes
    dangling_nodes = np.where(graph.sum(axis=1) == 0, 1, 0)

    # Iterate until convergence or max iterations
    for _ in range(max_iterations):
        # Compute new ranks
        new_ranks = (1 - d) / N + d * (graph @ (ranks / graph.sum(axis=1, where=~dangling_nodes.astype(bool))) + ranks[dangling_nodes].sum())

        # Compute change in ranks
        delta = np.abs(new_ranks - ranks).sum()

        # Check convergence
        if delta < tol:
            break

        # Update ranks
        ranks = new_ranks

    return ranks
def execute(adj_matrix):
    # Execute the page_rank function
    output = compute_PageRank(adj_matrix)
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

