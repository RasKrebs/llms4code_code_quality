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

def page_rank(adj_matrix: np.ndarray, d: float = 0.85, max_iterations: int = 100, tol: float = 1e-6) -> np.ndarray:
    """
    Compute PageRank for nodes in a directed graph represented by a numpy adjacency matrix.
    
    Parameters:
    adj_matrix (np.ndarray): The adjacency matrix representing the directed graph
    d (float): Damping factor (default 0.85)
    max_iterations (int): Maximum number of iterations (default 100)
    tol (float): Convergence tolerance (default 1e-6)
    
    Returns:
    np.ndarray: PageRank scores for each node
    
    Raises:
    ValueError: If the adjacency matrix is not square or contains invalid values
    """
    
    # Check if the adjacency matrix is square
    if adj_matrix.shape[0] != adj_matrix.shape[1]:
        raise ValueError("Adjacency matrix must be square")
    
    # Check if the adjacency matrix contains only 0s and 1s
    if np.any((adj_matrix != 0) & (adj_matrix != 1)):
        raise ValueError("Adjacency matrix can only contain 0s and 1s")
    
    # Normalize the adjacency matrix to ensure that each row sums to 1
    adj_matrix = adj_matrix / adj_matrix.sum(axis=1, keepdims=True)
    
    # Get the number of nodes
    n = adj_matrix.shape[0]
    
    # Initialize ranks equally
    ranks = np.ones(n) / n
    
    # Iterative computation of PageRank
    for _ in range(max_iterations):
        prev_ranks = ranks.copy()
        ranks = (1 - d) / n + d * np.dot(adj_matrix.T, ranks)
        
        # Check for convergence
        if np.linalg.norm(ranks - prev_ranks, ord=1) < tol:
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

