import numpy as np
import random

def page_rank(A, damping=0.85, n_iterations=10):
    """Calculate PageRank for a given directed graph."""
    # Initialize variables
    n_nodes = A.shape[0]  # Number of nodes in the graph
    pr_vector = np.zeros(n_nodes)  # PageRank vector, initially all zeros

    # Initialize the distribution vector to an uniform distribution
    distr_vector = (1 / n_nodes) * np.ones(n_nodes)  # Uniform distribution vector
    sink_rank = (1 - damping) / n_nodes  # Rank allocated to nodes with no outgoing links
    
    for iteration in range(n_iterations):
        distr_vector = np.zeros(n_nodes)
    
        for i in range(n_nodes):  # For each node i in the graph
            column_i = A[:, i]  # Get all incoming links to node i
    
            for j in range(n_nodes):  # For each node j with an incoming link from node i
                if column_i[j] == 1:  # If there's a link from node j to node i
                    pr_vector[i] += (damping * distr_vector[j]) / np.sum(column_i)
    
            pr_vector[i] += sink_rank  # Add sink rank if node i has no outgoing links
    
        distr_vector = pr_vector
    
    return pr_vector  # Return PageRank vector after n_iterations iterations
