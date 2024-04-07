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

def page_rank(adj_matrix, epochs=100, d=0.85):
    # Get total nodes from adjacency matrix
    n = len(adj_matrix)

    # Create a column vector with initial PageRank of 1/n for all nodes
    pr_old = np.ones((n, 1)) / n

    # Create a square matrix of shape (n, n) filled with ones
    outgoing_links = np.sum(adj_matrix, axis=0)
    link_matrix = adj_matrix / outgoing_links

    for _ in range(epochs):
        pr_new = d * np.matmul(link_matrix, pr_old) + (1 - d) / n

        # Check convergence i.e., check if our new values converge to our old ones
        if np.allclose(pr_new, pr_old):
            return pr_new.flatten()

        pr_old = pr_new

    print("Power iteration didn't converge in {} epochs".format(epochs))
    return pr_old.flatten()
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

