import numpy as np
from typing import Union

def page_rank(adjacency_matrix: Union[np.ndarray, list], d=0.85, max_iterations=100, tol=1e-6):
    """Computes PageRank for nodes in a directed graph represented by a numpy adjacency matrix.

    Parameters:
        adjacency_matrix (Union[np.ndarray, list]): The directed graph as an adjacency matrix where each element is either 0 or 1 representing whether there is an edge from i to j.
        d (float): Damping factor, defaults to 0.85.
        max_iterations (int): Maximum number of iterations in the PageRank computation, default is 100.
        tol (float): Convergence tolerance for PageRank values, default is 1e-6.

    Returns:
        page_rank(np.ndarray) : The computed page rank scores as a numpy array.
    """

    # Checking the input parameters and converting to numpy arrays if they are not already
    try:
        adjacency_matrix = np.asfarray(adjacency_matrix)
    except Exception as e:
        print("Error in Adjacency Matrix, please check the inputs")

    # Checking that the matrix is a square matrix
    if len(adjacency_matrix.shape) != 2 or adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
        print("Adjacency Matrix must be square")
        return None

    # Checking that the matrix is a directed graph (only non-negative entries allowed)
    if np.any(adjacency_matrix < 0):
        print("Adjacency Matrix should only contain non-negative values for directed graphs")
        return None

    # Initializing page rank scores, where each node starts with the same score (1/n)
    n = adjacency_matrix.shape[0]
    p = np.ones(n)/float(n)

    # Calculating the out-links for each node
    out_degree = adjacency_matrix.sum(axis=1)
    out_degree[out_degree==0] = 1 # For nodes without any outgoing edges, we consider them having a self loop
    teleportation_vector = np.ones((n,1))/float(n)

    for _ in range(max_iterations):
        new_p = (1-d)/n + d * np.dot(adjacency_matrix.T / out_degree, p) + d * teleportation_vector

        if np.sum((new_p - p)**2) < tol:  # Checking for convergence
            return new_p

        p = new_p

    print("Maximum iterations reached without reaching the desired tolerance")
    return p