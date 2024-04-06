import numpy as np

def page_rank(adj_matrix: np.ndarray, d: float = 0.85, max_iterations: int = 100, tol: float = 1e-6) -> np.ndarray:
    """
    Compute PageRank for nodes in a directed graph represented by a numpy adjacency matrix.
    
    Parameters:
    adj_matrix (np.ndarray): The adjacency matrix representing the directed graph
    d (float): Damping factor (default 0.85)
    max_iterations (int): Maximum number of iterations (default 100)
    tol (float): Convergence tolerance (default 1e-6)
    
    Returns:
    np.ndarray: PageRank scores for each node
    
    Raises:
    ValueError: If the adjacency matrix is not square or contains invalid values
    """
    
    # Check if the adjacency matrix is square
    if adj_matrix.shape[0] != adj_matrix.shape[1]:
        raise ValueError("Adjacency matrix must be square")
    
    # Check if the adjacency matrix contains only 0s and 1s
    if np.any((adj_matrix != 0) & (adj_matrix != 1)):
        raise ValueError("Adjacency matrix can only contain 0s and 1s")
    
    # Normalize the adjacency matrix to ensure that each row sums to 1
    adj_matrix = adj_matrix / adj_matrix.sum(axis=1, keepdims=True)
    
    # Get the number of nodes
    n = adj_matrix.shape[0]
    
    # Initialize ranks equally
    ranks = np.ones(n) / n
    
    # Iterative computation of PageRank
    for _ in range(max_iterations):
        prev_ranks = ranks.copy()
        ranks = (1 - d) / n + d * np.dot(adj_matrix.T, ranks)
        
        # Check for convergence
        if np.linalg.norm(ranks - prev_ranks, ord=1) < tol:
            break
    
    return ranks
