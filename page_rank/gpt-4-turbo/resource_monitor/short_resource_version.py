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

def page_rank(G, d=0.85, tol=1.0e-6):
    """
    Compute the PageRank of each node in a graph represented as an adjacency matrix.

    Parameters:
    G : numpy array
        Adjacency matrix of the graph where G[i, j] is a non-negative value representing an edge from node i to node j.
    d : float
        Damping factor. Represents the probability at each page the "random surfer" will get bored and request another random page. 
        Usually set to 0.85.
    tol : float
        Tolerance for convergence. The algorithm iterates until the norm of the difference in PageRank values between iterations is less than tol.

    Returns:
    numpy array
        Vector of PageRank values where each value corresponds to a node in the graph.
    """
    
    n = G.shape[0]  # Number of nodes
    M = G / G.sum(axis=0, keepdims=True)  # Normalized adjacency matrix (column-wise)
    M = np.nan_to_num(M)  # Replace NaNs with zero in case of columns with sum zero
    
    # Initial PageRank values
    r = np.ones(n) / n
    
    # Power iteration
    while True:
        r_new = d * M @ r + (1 - d) / n
        if np.linalg.norm(r_new - r, ord=1) < tol:
            break
        r = r_new
    
    return r

# Example usage
if __name__ == "__main__":
    # Create an example adjacency matrix
    G = np.array([[0, 1, 1, 0],
                  [1, 0, 1, 1],
                  [0, 0, 0, 1],
                  [0, 0, 1, 0]], dtype=float)
    
    ranks = page_rank(G)
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

