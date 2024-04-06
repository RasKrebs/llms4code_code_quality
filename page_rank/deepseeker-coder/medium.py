import numpy as np
from typing import List, Tuple

def page_rank(adjacency_matrix: np.ndarray, damping: float = 0.85, iterations: int = 100, tolerance: float = 1e-6) -> np.ndarray:
    """
    Compute the PageRank of the nodes in the adjacency matrix.

    Parameters:
    adjacency_matrix (np.ndarray): The adjacency matrix of the graph.
    damping (float): The damping factor, usually set to 0.85.
    iterations (int): The maximum number of iterations to run the algorithm.
    tolerance (float): The minimum difference for the algorithm to consider the PageRank values converged.

    Returns:
    np.ndarray: The PageRank values of the nodes in the adjacency matrix.
    """

    # Get the number of nodes
    num_nodes = adjacency_matrix.shape[0]

    # Initialize the PageRank values
    page_rank_values = np.ones(num_nodes) / num_nodes

    # Compute the out-degree of each node
    out_degrees = adjacency_matrix.sum(axis=1)

    # Handle dangling nodes by teleporting to all nodes
    teleportation_matrix = np.ones((num_nodes, num_nodes)) / num_nodes

    # Iterate until convergence or max iterations
    for _ in range(iterations):
        old_page_rank_values = page_rank_values.copy()

        # Update the PageRank values
        page_rank_values = damping * np.dot(adjacency_matrix, page_rank_values / out_degrees) + (1 - damping) * np.dot(teleportation_matrix, page_rank_values)

        # Check for convergence
        if np.linalg.norm(page_rank_values - old_page_rank_values, 1) < tolerance:
            break

    return page_rank_values
