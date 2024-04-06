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

def pagerank(A: np.ndarray, d: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """
    Computes PageRank for nodes in a directed graph represented by an adjacency matrix.

    Args:
        A: Adjacency matrix of the directed graph.
        d: Damping factor (default: 0.85).
        max_iter: Maximum number of iterations (default: 100).
        tol: Convergence tolerance (default: 1e-6).

    Returns:
        PageRank scores for each node in the graph.

    Raises:
        TypeError: If A is not a numpy array.
        ValueError: If A has negative elements, d is not in [0, 1], max_iter is not positive, or tol is not positive.
    """

    # Input validation
    if not isinstance(A, np.ndarray):
        raise TypeError("Adjacency matrix must be a numpy array")
    if not np.all(A >= 0):
        raise ValueError("Adjacency matrix elements must be non-negative")
    if d < 0 or d > 1:
        raise ValueError("Damping factor must be between 0 and 1")
    if max_iter <= 0:
        raise ValueError("Maximum iterations must be positive")
    if tol <= 0:
        raise ValueError("Convergence tolerance must be positive")

    n = A.shape[0]

    # Handle dangling nodes
    out_degree = np.sum(A, axis=1, dtype=bool)
    dangling = ~out_degree
    A[dangling, :] = 1 / n  # Distribute PageRank from dangling nodes evenly

    # Initialize PageRank vector
    PR = np.ones(n) / n

    # Iterate until convergence or max iterations reached
    for _ in range(max_iter):
        new_PR = d * A @ PR + (1 - d) / n  # Matrix-vector multiplication for efficiency
        delta = np.abs(new_PR - PR).sum()
        if delta < tol:
            break
        PR = new_PR

    return PR
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

