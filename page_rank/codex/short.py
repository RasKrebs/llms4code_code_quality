import numpy as np

def page_rank(adjacency_matrix, damping_factor=0.85, max_iterations=100, tol=1e-06):
    # Number of nodes
    num_nodes = adjacency_matrix.shape[0]

    # Normalize the adjacency matrix
    adjacency_matrix = adjacency_matrix / adjacency_matrix.sum(axis=1)[:, np.newaxis]

    # Initialize the PageRank vector
    page_rank_vector = np.ones(num_nodes) / num_nodes

    # Iterative algorithm
    for _ in range(max_iterations):
        prev_page_rank_vector = page_rank_vector
        page_rank_vector = (1.0 - damping_factor) / num_nodes + damping_factor * adjacency_matrix.T.dot(prev_page_rank_vector)
        diff = np.abs(prev_page_rank_vector - page_rank_vector).sum()
        if diff < tol:
            return page_rank_vector

    return page_rank_vector
