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

def page_rank(adj_matrix: np.ndarray, damping: float = 0.85, 
              max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """
    Calculates PageRank from an adjacency matrix.

    Args:
        adj_matrix: A square numpy array representing the adjacency matrix of the link structure.
        damping: The damping factor (between 0 and 1) for random jumps.
        max_iter: The maximum number of iterations allowed.
        tol: The tolerance for convergence (difference between PageRank estimates in consecutive iterations).

    Returns:
        A numpy array containing the PageRank values for each node.
    """

    n = adj_matrix.shape[0]
    out_degrees = np.sum(adj_matrix, axis=1, keepdims=True)
    dangling_nodes = np.where(out_degrees.flatten() == 0)[0]
    teleport = np.ones(n) / n

    # Handle dangling nodes with teleportation
    if len(dangling_nodes) > 0:
        teleport += np.sum(adj_matrix, axis=0) / n

    # Initialize PageRank vector with uniform distribution
    rank = np.ones(n) / n

    for _ in range(max_iter):
        new_rank = damping * (adj_matrix @ rank) + (1 - damping) * teleport
        delta = np.abs(new_rank - rank).sum()

        if delta <= tol:
            return new_rank

        rank = new_rank

        # Maximum iterations reached without convergence
        print("Warning: PageRank did not converge within tolerance.")
    return rank
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

