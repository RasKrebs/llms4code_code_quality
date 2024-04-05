
import numpy as np

def page_rank(G, d=0.85, tol=1.0e-6):
    """
    Compute the PageRank of each node in a graph represented as an adjacency matrix.

    Parameters:
    G : numpy array
        Adjacency matrix of the graph where G[i, j] is a non-negative value representing an edge from node i to node j.
    d : float
        Damping factor. Represents the probability at each page the "random surfer" will get bored and request another random page. 
        Usually set to 0.85.
    tol : float
        Tolerance for convergence. The algorithm iterates until the norm of the difference in PageRank values between iterations is less than tol.

    Returns:
    numpy array
        Vector of PageRank values where each value corresponds to a node in the graph.
    """
    
    n = G.shape[0]  # Number of nodes
    M = G / G.sum(axis=0, keepdims=True)  # Normalized adjacency matrix (column-wise)
    M = np.nan_to_num(M)  # Replace NaNs with zero in case of columns with sum zero
    
    # Initial PageRank values
    r = np.ones(n) / n
    
    # Power iteration
    while True:
        r_new = d * M @ r + (1 - d) / n
        if np.linalg.norm(r_new - r, ord=1) < tol:
            break
        r = r_new
    
    return r

# Example usage
if __name__ == "__main__":
    # Create an example adjacency matrix
    G = np.array([[0, 1, 1, 0],
                  [1, 0, 1, 1],
                  [0, 0, 0, 1],
                  [0, 0, 1, 0]], dtype=float)
    
    ranks = page_rank(G)
    print("PageRank values:", ranks)