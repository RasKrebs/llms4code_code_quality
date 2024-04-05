import numpy as np

def page_rank(adj_matrix, d=0.85, max_iter=100, tol=1e-6):
    # Number of nodes
    n = adj_matrix.shape[0]
    
    # Initialize PageRank values
    pr = np.ones(n) / n
    
    for i in range(max_iter):
        new_pr = np.zeros(n)
        
        for j in range(n):
            # Calculate the sum of incoming PageRank values for node j
            incoming_pr = np.sum(pr * adj_matrix[:, j])
            
            # Update the PageRank value for node j
            new_pr[j] = (1 - d) / n + d * incoming_pr
        
        # Check for convergence
        if np.linalg.norm(new_pr - pr, 1) < tol:
            break
        
        pr = new_pr
    
    return pr