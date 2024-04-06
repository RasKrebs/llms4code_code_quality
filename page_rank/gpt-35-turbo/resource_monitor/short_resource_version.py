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

def page_rank(adj_matrix, d=0.85, max_iter=100, tol=1e-6):
    # Number of nodes
    n = adj_matrix.shape[0]
    
    # Initialize PageRank values
    pr = np.ones(n) / n
    
    for i in range(max_iter):
        new_pr = np.zeros(n)
        
        for j in range(n):
            # Calculate the sum of incoming PageRank values for node j
            incoming_pr = np.sum(pr * adj_matrix[:, j])
            
            # Update the PageRank value for node j
            new_pr[j] = (1 - d) / n + d * incoming_pr
        
        # Check for convergence
        if np.linalg.norm(new_pr - pr, 1) < tol:
            break
        
        pr = new_pr
    
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

