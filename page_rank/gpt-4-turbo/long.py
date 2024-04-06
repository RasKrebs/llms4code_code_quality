import numpy as np

def compute_pagerank(graph: np.ndarray, d: float = 0.85, 
                     max_iterations: int = 100,
                     tol: float = 1e-6) -> np.ndarray:
    """
    Compute PageRank for a directed graph represented as an adjacency matrix.

    Parameters:
    - graph: np.ndarray. Adjacency matrix representation of the graph.
    - d: float. Damping factor, representing the probability to continue 
        browsing to another page. Defaults to 0.85.
    - max_iterations: int. Maximum number of iterations before halting the computation. 
        Defaults to 100.
    - tol: float. Convergence tolerance. Algorithm considers converged when the 
        change in PageRank values between iterations is under this value. Defaults to 1e-6.

    Returns:
    - np.ndarray. The PageRank of each node.

    Raises:
    - ValueError: If the input graph is not a valid adjacency matrix.
    """

    if not isinstance(graph, np.ndarray) or len(graph.shape) != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError("Input graph must be a square numpy array.")

    n = graph.shape[0]

    # Initialize PageRank values uniformly
    ranks = np.full(n, 1 / n)

    # Adjust the graph matrix for dangling nodes by adding a teleporting probability to all nodes
    dangling_weights = np.where(graph.sum(axis=1) == 0, 1 / n, 0)
    adjacency_matrix = graph / graph.sum(axis=1, where=(graph.sum(axis=1) != 0))[:, np.newaxis]

    for _ in range(max_iterations):
        prev_ranks = ranks.copy()

        # Calculate the rank contributions from linked pages
        rank_contributions = np.dot(adjacency_matrix.T, ranks)

        # Add the dangling weights and apply the damping factor
        ranks = d * (rank_contributions + dangling_weights * ranks.sum()) + (1 - d) / n

        if np.linalg.norm(ranks - prev_ranks, 1) < tol:
            break

    return ranks
