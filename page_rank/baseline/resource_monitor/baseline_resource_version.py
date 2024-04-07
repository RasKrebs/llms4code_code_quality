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

def page_rank(graph, damping=0.85, max_iterations=100, tol=1.0e-6):
    """
    Compute the PageRank of each node in the graph.

    Parameters:
    - graph: numpy.ndarray, a square adjacency matrix where graph[i, j] = 1 indicates
             a link from node i to node j, and graph[i, j] = 0 indicates no link.
    - damping: float, damping factor, usually set to 0.85.
    - max_iterations: int, the maximum number of iterations to perform.
    - tol: float, the tolerance to determine convergence. If the change in PageRank
           between iterations is less than this value, the algorithm will terminate.

    Returns:
    - ranks: numpy.ndarray, the PageRank of each node.
    """
    nodes = graph.shape[0]  # Number of nodes
    ranks = np.ones(nodes) / nodes  # Initialize PageRank values uniformly

    # Calculate the number of outbound links for each node
    outbound_links = graph.sum(axis=1)

    for _ in range(max_iterations):
        # Initialize new PageRank values with the teleportation term
        new_ranks = np.ones(nodes) * (1 - damping) / nodes

        # Calculate the PageRank for each node
        for i in range(nodes):
            for j in range(nodes):
                if graph[j, i]:  # If node j links to node i
                    # Add the rank contribution from node j to node i
                    new_ranks[i] += damping * ranks[j] / outbound_links[j]

        # Check if the algorithm has converged
        if np.linalg.norm(new_ranks - ranks, 1) < tol:
            break

        # Update the ranks for the next iteration
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

