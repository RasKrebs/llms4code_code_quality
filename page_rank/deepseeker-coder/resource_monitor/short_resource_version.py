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

def page_rank(adj_matrix, damping_factor=0.85, max_iterations=100, tol=1e-6):
    """
    Calculate the PageRank of a directed graph represented as a numpy adjacency matrix.

    Parameters:
    adj_matrix (numpy.ndarray): The adjacency matrix of the graph.
    damping_factor (float): The damping factor for the PageRank algorithm.
    max_iterations (int): The maximum number of iterations to run the algorithm.
    tol (float): The tolerance for the convergence of the algorithm.

    Returns:
    numpy.ndarray: The PageRank vector of the graph.
    """
    # Normalize the adjacency matrix
    out_degree = np.sum(adj_matrix, axis=0)
    out_degree[out_degree == 0] = 1  # Avoid division by zero
    adj_matrix = adj_matrix / out_degree

    # Initialize the PageRank vector
    num_nodes = adj_matrix.shape[0]
    pr = np.ones(num_nodes) / num_nodes

    # Iterate until convergence or max_iterations
    for _ in range(max_iterations):
        old_pr = pr.copy()
        pr = damping_factor * adj_matrix.dot(pr) + (1 - damping_factor) / num_nodes
        if np.linalg.norm(pr - old_pr, 1) < tol:
            break

    return pr
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

