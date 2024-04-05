import numpy as np
from typing import Union

def page_rank(graph: np.ndarray, d: float = 0.85, max_iterations: int = 100, tol: float = 1e-6) -> Union[np.ndarray, None]:
    """
    Compute PageRank for nodes in a directed graph represented by a numpy adjacency matrix.

    Parameters:
    graph (np.ndarray): A square numpy array representing the adjacency matrix of the graph.
    d (float): Damping factor, default is 0.85.
    max_iterations (int): Maximum number of iterations, default is 100.
    tol (float): Convergence tolerance, default is 1e-6.

    Returns:
    np.ndarray: A numpy array representing the PageRank of the nodes in the graph.
    None: If the input graph is not a square numpy array or if the graph contains negative values.
    """

    # Check if the input graph is a square numpy array
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        print("Error: The input graph must be a square numpy array.")
        return None

    # Check if the graph contains negative values
    if np.any(graph < 0):
        print("Error: The graph must not contain negative values.")
        return None

    # Initialize the ranks
    ranks = np.ones(graph.shape[0]) / graph.shape[0]

    # Iterate until convergence or reaching the max iterations
    for _ in range(max_iterations):
        old_ranks = ranks.copy()

        # Adjust for outbound links and the damping factor
        ranks = d * np.dot(graph, ranks)

        # Factor in teleportation for nodes without outbound links
        ranks += (1 - d) / graph.shape[0]

        # Check for convergence
        if np.linalg.norm(ranks - old_ranks) < tol:
            break

    return ranks