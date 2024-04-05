import numpy as np
from numpy import linalg as LA

def page_rank(M, num_iterations=100, d=0.85):
    """
    This function calculates the PageRank of a directed graph represented using an adjacency matrix M.

    Parameters:
        - M (numpy array) : Adjacency matrix representing the graph structure.
        - num_iterations (int): Number of iterations to run the power method for. Default is set to 100.
        - d (float): Probability that surfer randomly jumps to a new page each time they follow a link. Default is set to 0.85.

    Returns:
        - ranks (numpy array) : An array of PageRank scores for the input nodes in the graph.

    """
    N = M.shape[1] # Number of nodes in the graph
    v_init = np.random.rand(N, 1) # Initializing initial vector
    v_init = v_init / LA.norm(v_init, 1) # Normalizing the randomized vector to have unit length

    # Creating probability transition matrix
    M_hat = np.ones([N, N])/float(N) * d + (np.ones([N, N]) - M)/float(N) * (1-d)

    v = v_init
    for _ in range(num_iterations):
        v = np.matmul(M_hat, v)

    # Normalizing the final result so that it sums to one
    ranks = v/LA.norm(v, 1)

    return ranks