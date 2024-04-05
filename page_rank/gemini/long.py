import numpy as np

def pagerank(A: np.ndarray, d: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """
    Computes PageRank for nodes in a directed graph represented by an adjacency matrix.

    Args:
        A: Adjacency matrix of the directed graph.
        d: Damping factor (default: 0.85).
        max_iter: Maximum number of iterations (default: 100).
        tol: Convergence tolerance (default: 1e-6).

    Returns:
        PageRank scores for each node in the graph.

    Raises:
        TypeError: If A is not a numpy array.
        ValueError: If A has negative elements, d is not in [0, 1], max_iter is not positive, or tol is not positive.
    """

    # Input validation
    if not isinstance(A, np.ndarray):
        raise TypeError("Adjacency matrix must be a numpy array")
    if not np.all(A >= 0):
        raise ValueError("Adjacency matrix elements must be non-negative")
    if d < 0 or d > 1:
        raise ValueError("Damping factor must be between 0 and 1")
    if max_iter <= 0:
        raise ValueError("Maximum iterations must be positive")
    if tol <= 0:
        raise ValueError("Convergence tolerance must be positive")

    n = A.shape[0]

    # Handle dangling nodes
    out_degree = np.sum(A, axis=1, dtype=bool)
    dangling = ~out_degree
    A[dangling, :] = 1 / n  # Distribute PageRank from dangling nodes evenly

    # Initialize PageRank vector
    PR = np.ones(n) / n

    # Iterate until convergence or max iterations reached
    for _ in range(max_iter):
        new_PR = d * A @ PR + (1 - d) / n  # Matrix-vector multiplication for efficiency
        delta = np.abs(new_PR - PR).sum()
        if delta < tol:
            break
        PR = new_PR

    return PR
