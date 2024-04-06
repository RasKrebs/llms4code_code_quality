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

def page_rank(adj_matrix: np.ndarray, damping: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """
    Calculate PageRank given an adjacency matrix

    Parameters:
    adj_matrix (np.ndarray): Adjacency matrix where the entry in the i-th row and j-th column is 1 if there is a link from i to j, and 0 otherwise.
    damping (float): Damping factor, usually set to 0.85.
    max_iter (int): Maximum number of iterations.
    tol (float): Tolerance. The algorithm stops if the difference between PageRank values in two successive iterations is less than this.

    Returns:
    np.ndarray: PageRank values
    """

    # Number of nodes
    n = adj_matrix.shape[0]

    # Initialize PageRank values evenly
    ranks = np.full(n, 1.0 / n)

    # Teleportation factor
    teleport = (1.0 - damping) / n

    # Normalize the adjacency matrix
    outlinks = np.sum(adj_matrix, axis=0)
    adj_matrix = adj_matrix / outlinks

    for _ in range(max_iter):
        new_ranks = damping * np.dot(adj_matrix, ranks) + teleport

        # Check for convergence
        if np.linalg.norm(new_ranks - ranks) < tol:
            return new_ranks

        ranks = new_ranks

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

