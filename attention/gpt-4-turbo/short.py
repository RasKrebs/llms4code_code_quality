import numpy as np

class Attention:
    def __init__(self, embedding_size):
        self.embedding_size = embedding_size
        # Initialize weights for query, key, and value matrices
        self.W_query = np.random.rand(embedding_size, embedding_size)
        self.W_key = np.random.rand(embedding_size, embedding_size)
        self.W_value = np.random.rand(embedding_size, embedding_size)
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    
    def forward(self, x):
        """
        x: Input matrix with shape (embedding_size, sequence_length)
        """
        # Compute queries, keys, and values
        queries = np.dot(self.W_query, x)
        keys = np.dot(self.W_key, x)
        values = np.dot(self.W_value, x)

        # Compute attention scores
        # for simplicity, we use dot-product attention here
        attn_scores = np.dot(queries.T, keys)  # Shape: (sequence_length, sequence_length)

        # Apply softmax to get the attention weights
        attn_weights = self.softmax(attn_scores)  # Shape remains the same: (sequence_length, sequence_length)
        
        # Apply the weights to the values
        weighted_values = np.dot(attn_weights, values.T)  # Transpose to get the shape right (sequence_length, embedding_size)
        
        # Here, for simplicity, the output is just the weighted sum of values.
        # In practice, there might be additional operations.
        return weighted_values.T  # Transposing back to maintain consistency (embedding_size, sequence_length)
