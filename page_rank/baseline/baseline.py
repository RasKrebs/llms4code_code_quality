import numpy as np

def page_rank(graph, damping=0.85, max_iterations=100, tol=1.0e-6):
    """
    Compute the PageRank of each node in the graph.

    Parameters:
    - graph: numpy.ndarray, a square adjacency matrix where graph[i, j] = 1 indicates
             a link from node i to node j, and graph[i, j] = 0 indicates no link.
    - damping: float, damping factor, usually set to 0.85.
    - max_iterations: int, the maximum number of iterations to perform.
    - tol: float, the tolerance to determine convergence. If the change in PageRank
           between iterations is less than this value, the algorithm will terminate.

    Returns:
    - ranks: numpy.ndarray, the PageRank of each node.
    """
    nodes = graph.shape[0]  # Number of nodes
    ranks = np.ones(nodes) / nodes  # Initialize PageRank values uniformly

    # Calculate the number of outbound links for each node
    outbound_links = graph.sum(axis=1)

    for _ in range(max_iterations):
        # Initialize new PageRank values with the teleportation term
        new_ranks = np.ones(nodes) * (1 - damping) / nodes

        # Calculate the PageRank for each node
        for i in range(nodes):
            for j in range(nodes):
                if graph[j, i]:  # If node j links to node i
                    # Add the rank contribution from node j to node i
                    new_ranks[i] += damping * ranks[j] / outbound_links[j]

        # Check if the algorithm has converged
        if np.linalg.norm(new_ranks - ranks, 1) < tol:
            break

        # Update the ranks for the next iteration
        ranks = new_ranks

    return ranks
