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
def compute_pagerank(graph, d=0.85, max_iterations=100, tol=1e-6):
    """
    Computes PageRank for nodes in a directed graph represented by an adjacency matrix.

    Parameters:
    - graph (numpy array): Adjacency matrix representing the graph.
    - d (float, optional): Damping factor, default is 0.85.
    - max_iterations (int, optional): Maximum number of iterations, default is 100.
    - tol (float, optional): Convergence tolerance, default is 1e-6.

    Returns:
    - ranks (numpy array): PageRank values for each node in the graph.
    """
    # Number of nodes in the graph
    N = graph.shape[0]

    # Initialize rank vector with equal probabilities
    ranks = np.ones(N) / N

    # Compute outbound links and handle dangling nodes
    outbound_links = graph.sum(axis=1)
    mask = (outbound_links == 0)
    if mask.any():
        ranks[mask] = 1.0 / N  # teleportation for dangling nodes

    # Compute the initial PageRank vector
    pr_init = np.copy(ranks)

    # Perform Power Iteration until convergence or reaching max iterations
    for _ in range(max_iterations):
        ranks = d * graph.T @ ranks + (1 - d) / N  # apply damping factor and teleportation

        # Check convergence condition
        if np.linalg.norm(ranks - pr_init) < tol:
            break

        # Update the initial PageRank vector for next iteration
        np.copyto(pr_init, ranks)

    return ranks
def execute(adj_matrix):
    # Execute the page_rank function
    output = compute_pagerank(adj_matrix)
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

