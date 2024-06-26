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

def page_rank(A, alpha=0.85, tol=1e-8, max_iter=100):
    """
    Calculates the PageRank of a directed graph represented as a numpy adjacency matrix.

    Args:
        A: A numpy adjacency matrix representing the directed graph.
        alpha: The damping factor (default: 0.85).
        tol: The tolerance for convergence (default: 1e-8).
        max_iter: The maximum number of iterations (default: 100).

    Returns:
        A numpy array containing the PageRank scores for each node.
    """
    # Normalize the adjacency matrix
    A = A / A.sum(axis=1, keepdims=True)

    # Create a vector of uniform initial PageRanks
    PR = np.ones(A.shape[0]) / A.shape[0]

    # Iterate until convergence or max_iter is reached
    for _ in range(max_iter):
        new_PR = alpha * A.dot(PR) + (1 - alpha) * (1 / A.shape[0])
        diff = np.abs(new_PR - PR).sum()
        if diff < tol:
            return new_PR
        PR = new_PR

    # If convergence is not reached, return the last iteration
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
    output = execute(graph)


    # Stop the monitoring
    monitoring = False
    monitor_thread.join()

    print(max_resources_usage['cpu']), print(max_resources_usage['memory'] / (1024**2))

