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



def page_rank(graph : np.ndarray, damping_factor : float = 0.85, max_iterations : int = 100, tol : float = 1e-6) -> np.ndarray:
    """
    Computes PageRank for nodes in a directed graph represented by a NumPy adjacency matrix.

    Parameters:
        graph (np.ndarray): A numpy array representing the adjacency matrix of the graph.
        damping_factor (float): The probability to follow an outgoing link from the current node in the random walk. Defaults to 0.85.
        max_iterations (int): Maximum number of iterations to perform before stopping if convergence is not reached. Defaults to 100.
        tol (float): Tolerance for convergence, measured as the summed difference in ranks between consecutive iterations. Defaults to 1e-6.

    Returns:
        page_rank (np.ndarray): A numpy array containing the PageRank values of each node.

    Raises:
        ValueError: If the graph is not a valid adjacency matrix or has negative values.
        ValueError: If the damping factor is not between 0 and 1.
        ValueError: If max_iterations or tol are not positive numbers.
    """

    # Validate input parameters
    if not (np.issubdtype(graph.dtype, np.number) and np.all(np.isfinite(graph))):
        raise ValueError("Graph must be a valid adjacency matrix with finite values.")
    elif graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError("Graph must be a square (i.e., symmetric) adjacency matrix.")
    elif np.any(graph < 0):
        raise ValueError("Graph must not have negative values.")
    elif damping_factor <= 0 or damping_factor >= 1:
        raise ValueError("Damping factor must be between 0 and 1.")
    elif max_iterations <= 0 or tol <= 0:
        raise ValueError("max_iterations and tol must be positive numbers.")

    n_nodes = graph.shape[0]
    page_rank = np.full(n_nodes, 1 / n_nodes)  # Initialize ranks equally

    for _ in range(max_iterations):
        prev_page_rank = page_rank.copy()

        # Calculate outgoing links and the damping factor
        out_degree = np.where(graph > 0, graph, 1).sum(axis=1)
        weighted_out_links = np.divide(graph, out_degree[:, None]) * damping_factor

        # Calculate the teleportation probability for nodes without outbound links
        teleportation_prob = (1 - np.count_nonzero(weighted_out_links, axis=1) / n_nodes) * (1 - damping_factor)

        page_rank = weighted_out_links.sum(axis=0) + teleportation_prob  # Calculate the new ranks
        rank_diffs = np.abs(page_rank - prev_page_rank).sum()

        if rank_diffs < tol:  # Check for convergence
            break

    return page_rank  # Return PageRank as a numpy array
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

