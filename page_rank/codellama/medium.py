def page_rank(adjacency_matrix : numpy.ndarray, damping : float = 0.85, max_iterations : int = 100, tolerance : float = 1e-6) -> numpy.ndarray:
    """
    Calculates the PageRank of each node in a graph based on its adjacency matrix.

    Parameters
    ----------
    adjacency_matrix : numpy.ndarray
        The adjacency matrix representing the link structure between nodes. A value of 1 indicates a link, and 0 represents no link.
    damping : float, optional
        The damping factor for PageRank calculation (typically between 0.85 and 0.9). Defaults to 0.85.
    max_iterations : int, optional
        The maximum number of iterations to run before terminating if convergence is not reached. Defaults to 100.
    tolerance : float, optional
        A threshold for determining when the algorithm has converged. PageRank values should change by less than this amount between iterations. Defaults to 1e-6.

    Returns
    -------
    numpy.ndarray
        The PageRank value of each node in the graph, as an array with shape (n_nodes,).
    """

    # Validate input parameters
    if not isinstance(adjacency_matrix, numpy.ndarray):
        raise TypeError("Adjacency matrix must be a NumPy ndarray.")
    if adjacency_matrix.dtype != bool:
        raise ValueError("Adjacency matrix values must be boolean (1 represents a link, and 0 represents no link).")

    n_nodes = len(adjacency_matrix)
    rank_array = numpy.full((n_nodes,), 1 / n_nodes)  # Initialize ranks evenly

    for iteration in range(max_iterations):
        new_rank_array = (1 - damping) / n_nodes + damping * adjacency_matrix.T @ rank_array

        if numpy.linalg.norm(new_rank_array - rank_array) < tolerance:  # Convergence reached
            break

        rank_array = new_rank_array

    return rank_array

