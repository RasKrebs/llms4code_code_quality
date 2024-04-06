import numpy as np
from typing import Union

def page_rank(adjacency_matrix: np.ndarray, d: float = 0.85, iterations: int = 100, tol: float = 1e-3) -> np.ndarray:
    """
    Computes the PageRank of nodes in a graph given by its adjacency matrix.

    Parameters:
        adjacency_matrix (np.ndarray): The adjacency matrix of a graph, where entry ij represents whether there exists an edge from node i to node j.
        d (float): Damping factor for PageRank.  Default is 0.85.
        iterations (int): Maximum number of iterations for the algorithm to converge. Default is 100.
        tol (float): Minimal difference between PageRanks in consecutive iterations to consider the computation converged. Default is 1e-3.

    Returns:
        np.ndarray: The PageRank of each node in the graph.
    """
    N = len(adjacency_matrix) # Number of nodes

    # Step 1: Initialization
    ranks = np.ones(N) / N # Initially, all nodes have the same rank (1/n)
    teleportation = np.logical_not(np.any(adjacency_matrix, axis=0)) # Teleports to all nodes for dangling nodes

    # Step 2: Iteratively update PageRank until convergence or max iterations is reached
    for _ in range(iterations):
        old_ranks = ranks.copy()

        # Update PageRanks based on the damping factor, link structure and teleportation
        ranks = d * adjacency_matrix @ ranks + (1 - d) / N
        ranks += (d * (teleportation @ ranks)) / np.sum(teleportation)

        # Check for convergence
        if np.linalg.norm(ranks - old_ranks, 1) < tol:
            break

    return ranks
