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
import random

def page_rank(A, damping=0.85, n_iterations=10):
    """Calculate PageRank for a given directed graph."""
    # Initialize variables
    n_nodes = A.shape[0]  # Number of nodes in the graph
    pr_vector = np.zeros(n_nodes)  # PageRank vector, initially all zeros

    # Initialize the distribution vector to an uniform distribution
    distr_vector = (1 / n_nodes) * np.ones(n_nodes)  # Uniform distribution vector
    sink_rank = (1 - damping) / n_nodes  # Rank allocated to nodes with no outgoing links
    
    for iteration in range(n_iterations):
        distr_vector = np.zeros(n_nodes)
    
        for i in range(n_nodes):  # For each node i in the graph
            column_i = A[:, i]  # Get all incoming links to node i
    
            for j in range(n_nodes):  # For each node j with an incoming link from node i
                if column_i[j] == 1:  # If there's a link from node j to node i
                    pr_vector[i] += (damping * distr_vector[j]) / np.sum(column_i)
    
            pr_vector[i] += sink_rank  # Add sink rank if node i has no outgoing links
    
        distr_vector = pr_vector
    
    return pr_vector  # Return PageRank vector after n_iterations iterations
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

