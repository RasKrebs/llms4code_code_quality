import numpy as np
def compute_pagerank(graph, d=0.85, max_iterations=100, tol=1e-6):
    """
    Computes PageRank for nodes in a directed graph represented by an adjacency matrix.

    Parameters:
    - graph (numpy array): Adjacency matrix representing the graph.
    - d (float, optional): Damping factor, default is 0.85.
    - max_iterations (int, optional): Maximum number of iterations, default is 100.
    - tol (float, optional): Convergence tolerance, default is 1e-6.

    Returns:
    - ranks (numpy array): PageRank values for each node in the graph.
    """
    # Number of nodes in the graph
    N = graph.shape[0]

    # Initialize rank vector with equal probabilities
    ranks = np.ones(N) / N

    # Compute outbound links and handle dangling nodes
    outbound_links = graph.sum(axis=1)
    mask = (outbound_links == 0)
    if mask.any():
        ranks[mask] = 1.0 / N  # teleportation for dangling nodes

    # Compute the initial PageRank vector
    pr_init = np.copy(ranks)

    # Perform Power Iteration until convergence or reaching max iterations
    for _ in range(max_iterations):
        ranks = d * graph.T @ ranks + (1 - d) / N  # apply damping factor and teleportation

        # Check convergence condition
        if np.linalg.norm(ranks - pr_init) < tol:
            break

        # Update the initial PageRank vector for next iteration
        np.copyto(pr_init, ranks)

    return ranks
