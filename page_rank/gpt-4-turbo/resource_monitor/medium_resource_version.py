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

def page_rank(adj_matrix: np.ndarray, damping: float = 0.85, iterations: int = 100, tolerance: float = 1.0e-6) -> np.ndarray:
    """
    Calculate PageRank given an adjacency matrix.

    Parameters:
    - adj_matrix (np.ndarray): Non-negative adjacency matrix where M[i, j] represents
                               the link from 'j' to 'i', i.e., row-stochastic form.
    - damping (float): Damping parameter for PageRank, usually set to 0.85.
    - iterations (int): Maximum number of iterations to perform.
    - tolerance (float): Tolerance to achieve before terminating the iterations.

    Returns:
    - np.ndarray: Vector of PageRank scores for each node.

    Note:
    This implementation handles dangling nodes by redistributing their rank uniformly.
    """
    n = adj_matrix.shape[0]
    
    # Convert adjacency matrix to probability matrix.
    out_degree = adj_matrix.sum(axis=0)
    prob_matrix = np.where(out_degree > 0, adj_matrix / out_degree, 0)
    
    # Handle dangling nodes (those with no outbound links).
    dangling_weights = np.where(out_degree == 0, 1.0 / n, 0)
    damping_value = (1.0 - damping) / n
    
    # Initialize ranks evenly.
    ranks = np.ones(n) / n
    
    for _ in range(iterations):
        # Calculate new ranks with damping factor and teleportation for dangling nodes.
        new_ranks = damping * np.dot(prob_matrix, ranks) + damping * np.dot(dangling_weights, ranks) + damping_value
        total_diff = np.abs(new_ranks - ranks).sum()
        
        if total_diff < tolerance:
            break
        
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
    output = execute(adj_matrix)


    # Stop the monitoring
    monitoring = False
    monitor_thread.join()

    print(max_resources_usage['cpu']), print(max_resources_usage['memory'] / (1024**2))

