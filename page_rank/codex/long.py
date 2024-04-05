import numpy as np

def compute_PageRank(graph: np.ndarray, d: float = 0.85, max_iterations: int = 100, tol: float = 1e-6) -> np.ndarray:
    """
    Compute PageRank for nodes in a directed graph represented by a numpy adjacency matrix.

    Parameters:
    graph: numpy adjacency matrix representing the graph.
    d: damping factor (default 0.85).
    max_iterations: maximum number of iterations (default 100).
    tol: convergence tolerance (default 1e-6).

    Returns:
    PageRank as a numpy array.
    """
    # Check input parameters
    if not 0 < d < 1:
        raise ValueError("Damping factor must be between 0 and 1.")
    if not max_iterations > 0:
        raise ValueError("Max iterations must be positive.")
    if not tol > 0:
        raise ValueError("Tolerance must be positive.")

    # Number of nodes
    N = graph.shape[0]

    # Initialize ranks
    ranks = np.full(N, 1/N)

    # Dangling nodes
    dangling_nodes = np.where(graph.sum(axis=1) == 0, 1, 0)

    # Iterate until convergence or max iterations
    for _ in range(max_iterations):
        # Compute new ranks
        new_ranks = (1 - d) / N + d * (graph @ (ranks / graph.sum(axis=1, where=~dangling_nodes)) + ranks[dangling_nodes].sum())

        # Compute change in ranks
        delta = np.abs(new_ranks - ranks).sum()

        # Check convergence
        if delta < tol:
            break

        # Update ranks
        ranks = new_ranks

    return ranks
