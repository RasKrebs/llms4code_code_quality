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
from numpy import linalg as LA

def page_rank(M, num_iterations=100, d=0.85):
    """
    This function calculates the PageRank of a directed graph represented using an adjacency matrix M.

    Parameters:
        - M (numpy array) : Adjacency matrix representing the graph structure.
        - num_iterations (int): Number of iterations to run the power method for. Default is set to 100.
        - d (float): Probability that surfer randomly jumps to a new page each time they follow a link. Default is set to 0.85.

    Returns:
        - ranks (numpy array) : An array of PageRank scores for the input nodes in the graph.

    """
    N = M.shape[1] # Number of nodes in the graph
    v_init = np.random.rand(N, 1) # Initializing initial vector
    v_init = v_init / LA.norm(v_init, 1) # Normalizing the randomized vector to have unit length

    # Creating probability transition matrix
    M_hat = np.ones([N, N])/float(N) * d + (np.ones([N, N]) - M)/float(N) * (1-d)

    v = v_init
    for _ in range(num_iterations):
        v = np.matmul(M_hat, v)

    # Normalizing the final result so that it sums to one
    ranks = v/LA.norm(v, 1)

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

