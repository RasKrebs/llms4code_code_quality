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

def page_rank(adjacency_matrix, damping=0.85, iterations=100, tolerance=1e-6):
    """
    Calculate PageRank from an adjacency matrix.

    Parameters:
    adjacency_matrix (np.ndarray): Adjacency matrix representing the link structure of the network.
    damping (float): Damping factor for the PageRank algorithm. Default is 0.85.
    iterations (int): Maximum number of iterations to calculate PageRank values. Default is 100.
    tolerance (float): Tolerance value for checking convergence. Default is 1e-6.

    Returns:
    ranks (np.ndarray): Array containing the PageRank values for each node in the network.
    """

    # Calculate the outdegree of each node
    outdegrees = np.sum(adjacency_matrix, axis=1)

    # Initialize rank vector with equal ranks
    num_nodes = adjacency_matrix.shape[0]
    ranks = np.ones(num_nodes) / num_nodes

    # Iterate until convergence or max iterations
    for _ in range(iterations):
        new_ranks = np.zeros(num_nodes)

        for i in range(num_nodes):
            if outdegrees[i] > 0:
                # Calculate the sum of ranks from nodes linking to node i
                rank_sum = np.sum((adjacency_matrix[:, i] * ranks) / outdegrees)

                # Update new rank for node i using damping and link structure
                new_ranks[i] = (1 - damping) + damping * rank_sum
            else:
                # Handle dangling nodes by teleportation
                new_ranks[i] = 0.5

        # Check convergence
        if np.linalg.norm(new_ranks - ranks) < tolerance:
            break

        # Update ranks for next iteration
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

