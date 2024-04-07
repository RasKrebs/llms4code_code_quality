import numpy as np

class Attention:
    """
    A simple implementation of the Attention mechanism with Scaled Dot-Product Attention.
    This class initializes random weight matrices for query, key, value, and output projections.
    """

    def __init__(self, emb_size: int):
        """
        Initializes the Attention mechanism.

        Parameters:
        - emb_size (int): The size of the embedding vectors.
        """
        self.emb_size = emb_size
        # Initialize weight matrices for query, key, value, and output projection
        self.W_query = np.random.rand(emb_size, emb_size)
        self.W_key = np.random.rand(emb_size, emb_size)
        self.W_value = np.random.rand(emb_size, emb_size)
        self.W_out = np.random.rand(emb_size, emb_size)

    def scaled_dot_product_attention(self, query: np.ndarray, key: np.ndarray, value: np.ndarray) -> np.ndarray:
        """
        Computes the Scaled Dot-Product Attention.

        Parameters:
        - query, key, value (np.ndarray): Matrices containing the query, key, and value vectors.

        Returns:
        - np.ndarray: The result of applying the attention mechanism.
        """
        d_k = query.shape[1]
        scores = np.matmul(query, key.T) / np.sqrt(d_k)  # Scaled scores
        attn_weights = self.softmax(scores)
        return np.matmul(attn_weights, value)

    def softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the softmax of the input array.

        Parameters:
        - x (np.ndarray): The input array.

        Returns:
        - np.ndarray: Softmax applied to the input array.
        """
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass of the Attention mechanism.

        Parameters:
        - x (np.ndarray): Input matrix of shape (embedding_size, sequence_length).

        Returns:
        - np.ndarray: Output of the Attention mechanism.
        """
        # Project inputs to query/key/value vectors
        queries = np.dot(x, self.W_query)
        keys = np.dot(x, self.W_key)
        values = np.dot(x, self.W_value)

        # Apply Scaled Dot-Product Attention
        attention_out = self.scaled_dot_product_attention(queries, keys, values)

        # Project the output of the attention
        out = np.dot(attention_out, self.W_out)

        return out
