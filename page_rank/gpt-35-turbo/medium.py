import numpy as np
from typing import List

def page_rank(adjacency_matrix: np.ndarray, damping: float = 0.85, iterations: int = 100, tolerance: float = 1e-6) -> np.ndarray:
    """
    Calculate PageRank from an adjacency matrix.
    
    Args:
    adjacency_matrix (np.ndarray): Adjacency matrix of the graph
    damping (float): Damping factor (default is 0.85)
    iterations (int): Maximum number of iterations (default is 100)
    tolerance (float): Convergence tolerance (default is 1e-6)
    
    Returns:
    np.ndarray: PageRank values for each node
    """
    
    # Get number of nodes
    n = adjacency_matrix.shape[0]
    
    # Initialize ranks evenly
    ranks = np.full((n,), 1/n)
    
    # Handling teleportation for dangling nodes
    dangling_nodes = np.where(adjacency_matrix.sum(axis=1) == 0)[0]
    
    # Iteration until convergence or max iterations
    for _ in range(iterations):
        prev_ranks = ranks.copy()
        
        for i in range(n):
            # Calculate contribution from incoming links
            incoming_ranks = ranks * adjacency_matrix[:, i]
            incoming_sum = incoming_ranks.sum()
            
            # Handle dangling nodes
            if i in dangling_nodes:
                incoming_sum = ranks[dangling_nodes].sum() / n
                
            # Update rank using damping factor
            ranks[i] = (1 - damping) / n + damping * incoming_sum
        
        # Check for convergence
        if np.linalg.norm(ranks - prev_ranks) < tolerance:
            break
    
    return ranks
