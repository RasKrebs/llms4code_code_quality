import numpy as np

def page_rank(adj_matrix, epochs=100, d=0.85):
    # Get total nodes from adjacency matrix
    n = len(adj_matrix)

    # Create a column vector with initial PageRank of 1/n for all nodes
    pr_old = np.ones((n, 1)) / n

    # Create a square matrix of shape (n, n) filled with ones
    outgoing_links = np.sum(adj_matrix, axis=0)
    link_matrix = adj_matrix / outgoing_links

    for _ in range(epochs):
        pr_new = d * np.matmul(link_matrix, pr_old) + (1 - d) / n

        # Check convergence i.e., check if our new values converge to our old ones
        if np.allclose(pr_new, pr_old):
            return pr_new.flatten()

        pr_old = pr_new

    print("Power iteration didn't converge in {} epochs".format(epochs))
    return pr_old.flatten()