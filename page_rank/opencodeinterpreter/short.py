import numpy as np

def page_rank(adjacency_matrix, d=0.85, max_iterations=100):
    n = adjacency_matrix.shape[0]

    # Ensure matrix is stochastic
    row_sums = adjacency_matrix.sum(axis=1)
    adjacency_matrix /= row_sums[:, np.newaxis]

    # Initialize PageRank vector
    pageranks = np.ones(n) / n

    for iteration in range(max_iterations):
        new_pageranks = (1 - d) / n + d * adjacency_matrix @ pageranks

        # Check for convergence
        if np.allclose(new_pageranks, pageranks, atol=0.001):
            return new_pageranks

        pageranks = new_pageranks

    return pageranks
