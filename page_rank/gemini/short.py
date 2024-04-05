import numpy as np

def page_rank(A, alpha=0.85, tol=1e-8, max_iter=100):
    """
    Calculates the PageRank of a directed graph represented as a numpy adjacency matrix.

    Args:
        A: A numpy adjacency matrix representing the directed graph.
        alpha: The damping factor (default: 0.85).
        tol: The tolerance for convergence (default: 1e-8).
        max_iter: The maximum number of iterations (default: 100).

    Returns:
        A numpy array containing the PageRank scores for each node.
    """
    # Normalize the adjacency matrix
    A = A / A.sum(axis=1, keepdims=True)

    # Create a vector of uniform initial PageRanks
    PR = np.ones(A.shape[0]) / A.shape[0]

    # Iterate until convergence or max_iter is reached
    for _ in range(max_iter):
        new_PR = alpha * A.dot(PR) + (1 - alpha) * (1 / A.shape[0])
        diff = np.abs(new_PR - PR).sum()
        if diff < tol:
            return new_PR
        PR = new_PR

    # If convergence is not reached, return the last iteration
    return PR
