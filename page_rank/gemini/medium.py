import numpy as np

def page_rank(adj_matrix: np.ndarray, damping: float = 0.85, 
              max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """
    Calculates PageRank from an adjacency matrix.

    Args:
        adj_matrix: A square numpy array representing the adjacency matrix of the link structure.
        damping: The damping factor (between 0 and 1) for random jumps.
        max_iter: The maximum number of iterations allowed.
        tol: The tolerance for convergence (difference between PageRank estimates in consecutive iterations).

    Returns:
        A numpy array containing the PageRank values for each node.
    """

    n = adj_matrix.shape[0]
    out_degrees = np.sum(adj_matrix, axis=1, keepdims=True)
    dangling_nodes = np.where(out_degrees.flatten() == 0)[0]
    teleport = np.ones(n) / n

    # Handle dangling nodes with teleportation
    if len(dangling_nodes) > 0:
        teleport += np.sum(adj_matrix, axis=0) / n

    # Initialize PageRank vector with uniform distribution
    rank = np.ones(n) / n

    for _ in range(max_iter):
        new_rank = damping * (adj_matrix @ rank) + (1 - damping) * teleport
        delta = np.abs(new_rank - rank).sum()

        if delta <= tol:
            return new_rank

        rank = new_rank

        # Maximum iterations reached without convergence
        print("Warning: PageRank did not converge within tolerance.")
    return rank
