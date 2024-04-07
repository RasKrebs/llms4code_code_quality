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
from typing import List

def page_rank(adjacency_matrix: np.ndarray, damping: float = 0.85, iterations: int = 100, tolerance: float = 1e-6) -> np.ndarray:
    """
    Calculate PageRank from an adjacency matrix.
    
    Args:
    adjacency_matrix (np.ndarray): Adjacency matrix of the graph
    damping (float): Damping factor (default is 0.85)
    iterations (int): Maximum number of iterations (default is 100)
    tolerance (float): Convergence tolerance (default is 1e-6)
    
    Returns:
    np.ndarray: PageRank values for each node
    """
    
    # Get number of nodes
    n = adjacency_matrix.shape[0]
    
    # Initialize ranks evenly
    ranks = np.full((n,), 1/n)
    
    # Handling teleportation for dangling nodes
    dangling_nodes = np.where(adjacency_matrix.sum(axis=1) == 0)[0]
    
    # Iteration until convergence or max iterations
    for _ in range(iterations):
        prev_ranks = ranks.copy()
        
        for i in range(n):
            # Calculate contribution from incoming links
            incoming_ranks = ranks * adjacency_matrix[:, i]
            incoming_sum = incoming_ranks.sum()
            
            # Handle dangling nodes
            if i in dangling_nodes:
                incoming_sum = ranks[dangling_nodes].sum() / n
                
            # Update rank using damping factor
            ranks[i] = (1 - damping) / n + damping * incoming_sum
        
        # Check for convergence
        if np.linalg.norm(ranks - prev_ranks) < tolerance:
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

