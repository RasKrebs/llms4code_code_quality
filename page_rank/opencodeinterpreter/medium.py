import numpy as np

def page_rank(adjacency_matrix, damping=0.85, iterations=100, tolerance=1e-6):
    """
    Calculate PageRank from an adjacency matrix.

    Parameters:
    adjacency_matrix (np.ndarray): Adjacency matrix representing the link structure of the network.
    damping (float): Damping factor for the PageRank algorithm. Default is 0.85.
    iterations (int): Maximum number of iterations to calculate PageRank values. Default is 100.
    tolerance (float): Tolerance value for checking convergence. Default is 1e-6.

    Returns:
    ranks (np.ndarray): Array containing the PageRank values for each node in the network.
    """

    # Calculate the outdegree of each node
    outdegrees = np.sum(adjacency_matrix, axis=1)

    # Initialize rank vector with equal ranks
    num_nodes = adjacency_matrix.shape[0]
    ranks = np.ones(num_nodes) / num_nodes

    # Iterate until convergence or max iterations
    for _ in range(iterations):
        new_ranks = np.zeros(num_nodes)

        for i in range(num_nodes):
            if outdegrees[i] > 0:
                # Calculate the sum of ranks from nodes linking to node i
                rank_sum = np.sum((adjacency_matrix[:, i] * ranks) / outdegrees)

                # Update new rank for node i using damping and link structure
                new_ranks[i] = (1 - damping) + damping * rank_sum
            else:
                # Handle dangling nodes by teleportation
                new_ranks[i] = 0.5

        # Check convergence
        if np.linalg.norm(new_ranks - ranks) < tolerance:
            break

        # Update ranks for next iteration
        ranks = new_ranks

    return ranks
