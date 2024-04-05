import numpy as np

def page_rank(adj_matrix: np.ndarray, damping: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """
    Calculate PageRank given an adjacency matrix

    Parameters:
    adj_matrix (np.ndarray): Adjacency matrix where the entry in the i-th row and j-th column is 1 if there is a link from i to j, and 0 otherwise.
    damping (float): Damping factor, usually set to 0.85.
    max_iter (int): Maximum number of iterations.
    tol (float): Tolerance. The algorithm stops if the difference between PageRank values in two successive iterations is less than this.

    Returns:
    np.ndarray: PageRank values
    """

    # Number of nodes
    n = adj_matrix.shape[0]

    # Initialize PageRank values evenly
    ranks = np.full(n, 1.0 / n)

    # Teleportation factor
    teleport = (1.0 - damping) / n

    # Normalize the adjacency matrix
    outlinks = np.sum(adj_matrix, axis=0)
    adj_matrix = adj_matrix / outlinks

    for _ in range(max_iter):
        new_ranks = damping * np.dot(adj_matrix, ranks) + teleport

        # Check for convergence
        if np.linalg.norm(new_ranks - ranks) < tol:
            return new_ranks

        ranks = new_ranks

    return ranks
