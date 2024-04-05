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