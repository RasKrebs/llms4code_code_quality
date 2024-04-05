def page_rank(graph : np.ndarray, damping_factor : float = 0.85, max_iterations : int = 100, tol : float = 1e-6) -> np.ndarray:
    """
    Computes PageRank for nodes in a directed graph represented by a NumPy adjacency matrix.

    Parameters:
        graph (np.ndarray): A numpy array representing the adjacency matrix of the graph.
        damping_factor (float): The probability to follow an outgoing link from the current node in the random walk. Defaults to 0.85.
        max_iterations (int): Maximum number of iterations to perform before stopping if convergence is not reached. Defaults to 100.
        tol (float): Tolerance for convergence, measured as the summed difference in ranks between consecutive iterations. Defaults to 1e-6.

    Returns:
        page_rank (np.ndarray): A numpy array containing the PageRank values of each node.

    Raises:
        ValueError: If the graph is not a valid adjacency matrix or has negative values.
        ValueError: If the damping factor is not between 0 and 1.
        ValueError: If max_iterations or tol are not positive numbers.
    """

    # Validate input parameters
    if not (np.issubdtype(graph.dtype, np.number) and np.all(np.isfinite(graph))):
        raise ValueError("Graph must be a valid adjacency matrix with finite values.")
    elif graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError("Graph must be a square (i.e., symmetric) adjacency matrix.")
    elif np.any(graph < 0):
        raise ValueError("Graph must not have negative values.")
    elif damping_factor <= 0 or damping_factor >= 1:
        raise ValueError("Damping factor must be between 0 and 1.")
    elif max_iterations <= 0 or tol <= 0:
        raise ValueError("max_iterations and tol must be positive numbers.")

    n_nodes = graph.shape[0]
    page_rank = np.full(n_nodes, 1 / n_nodes)  # Initialize ranks equally

    for _ in range(max_iterations):
        prev_page_rank = page_rank.copy()

        # Calculate outgoing links and the damping factor
        out_degree = np.where(graph > 0, graph, 1).sum(axis=1)
        weighted_out_links = np.divide(graph, out_degree[:, None]) * damping_factor

        # Calculate the teleportation probability for nodes without outbound links
        teleportation_prob = (1 - np.count_nonzero(weighted_out_links, axis=1) / n_nodes) * (1 - damping_factor)

        page_rank = weighted_out_links.sum(axis=0) + teleportation_prob  # Calculate the new ranks
        rank_diffs = np.abs(page_rank - prev_page_rank).sum()

        if rank_diffs < tol:  # Check for convergence
            break

    return page_rank  # Return PageRank as a numpy array