import numpy as np
from typing import Optional, Tuple

def page_rank(graph: np.ndarray, d: float = 0.85, max_iterations: int = 100, tol: float = 1e-6) -> Tuple[np.ndarray, int]:
    """
    Computes PageRank for nodes in a directed graph represented by an adjacency matrix.

    Parameters:
        graph (numpy array): A square numpy array where entry [i, j] is the number of links from node i to node j.

        d (float, optional): Damping factor. Defaults to 0.85.

        max_iterations (int, optional): Maximum number of iterations. Defaults to 100.

        tol (float, optional): Tolerance for convergence. Defaults to 1e-6.

    Returns:
        Tuple[np.ndarray, int]: A tuple containing the PageRank vector and the number of iterations performed.
    """

    # Ensure input is a square numpy array
    assert graph.shape[0] == graph.shape[1], "Adjacency matrix should be square"

    # Initialize ranks to 1/n for all nodes
    n = graph.shape[0]
    ranks = np.ones(n) / n

    out_links = graph.sum(axis=1).reshape(-1, 1)
    dangling_nodes = (out_links == 0).flatten()

    # Handle dangling nodes and normalize the adjacency matrix
    M = np.divide(graph, out_links, out=np.zeros_like(graph), where=(out_links != 0))
    M[:, dangling_nodes] = 1 / n

    # Apply damping factor and teleportation
    M *= d
    M += (1 - d) / n

    for i in range(max_iterations):
        old_ranks = ranks.copy()

        # Iterate PageRank equation
        ranks = np.dot(M, ranks)

        if np.linalg.norm(ranks - old_ranks) < tol:
            break

    return ranks, i+1
