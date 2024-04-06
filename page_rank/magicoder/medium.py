import numpy as np
from typing import Optional, List

def page_rank(matrix: np.ndarray, damping: float = 0.85, iterations: int = 100, tolerance: float = 0.0001) -> np.ndarray:
    """
    Calculate PageRank from an adjacency matrix with a specified number of iterations and a tolerance level for the convergence.

    Parameters:
        matrix (np.ndarray): A numpy array representing the adjacency matrix of the webgraph.
        damping (float): The probability that user will follow one outbound link in the matrix, default is 0.85.
        iterations (int): The number of iterations to run PageRank algorithm, default is 100.
        tolerance (float): The minimal change in pageranks required for a new iteration to start, default is 0.0001.

    Returns:
        np.ndarray: A numpy array representing the calculated pagerank vector with the same order as the nodes in the graph.
    """
    # Check if matrix is valid adjacency matrix
    assert(matrix.shape[0] == matrix.shape[1])

    # Number of Nodes
    N = len(matrix)

    # Initializing PageRank Vector
    pr_vector = np.full((N, 1), fill_value=1/N).astype('float64')

    for _ in range(iterations):
        new_pr_vector = (1 - damping)/N + damping * np.dot(matrix, pr_vector)

        if np.sum(np.abs(new_pr_vector - pr_vector)) <= tolerance:
            return new_pr_vector

        pr_vector = new_pr_vector

    # If max iterations reached without convergence
    return pr_vector
