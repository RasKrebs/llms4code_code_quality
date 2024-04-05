import numpy as np

def page_rank(adj_matrix: np.ndarray, damping: float = 0.85, iterations: int = 100, tolerance: float = 1.0e-6) -> np.ndarray:
    """
    Calculate PageRank given an adjacency matrix.

    Parameters:
    - adj_matrix (np.ndarray): Non-negative adjacency matrix where M[i, j] represents
                               the link from 'j' to 'i', i.e., row-stochastic form.
    - damping (float): Damping parameter for PageRank, usually set to 0.85.
    - iterations (int): Maximum number of iterations to perform.
    - tolerance (float): Tolerance to achieve before terminating the iterations.

    Returns:
    - np.ndarray: Vector of PageRank scores for each node.

    Note:
    This implementation handles dangling nodes by redistributing their rank uniformly.
    """
    n = adj_matrix.shape[0]
    
    # Convert adjacency matrix to probability matrix.
    out_degree = adj_matrix.sum(axis=0)
    prob_matrix = np.where(out_degree > 0, adj_matrix / out_degree, 0)
    
    # Handle dangling nodes (those with no outbound links).
    dangling_weights = np.where(out_degree == 0, 1.0 / n, 0)
    damping_value = (1.0 - damping) / n
    
    # Initialize ranks evenly.
    ranks = np.ones(n) / n
    
    for _ in range(iterations):
        # Calculate new ranks with damping factor and teleportation for dangling nodes.
        new_ranks = damping * np.dot(prob_matrix, ranks) + damping * np.dot(dangling_weights, ranks) + damping_value
        total_diff = np.abs(new_ranks - ranks).sum()
        
        if total_diff < tolerance:
            break
        
        ranks = new_ranks
    
    return ranks