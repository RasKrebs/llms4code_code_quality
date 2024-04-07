import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

class Attention:
    """
    Implements a simplistic Attention mechanism for educational purposes.
    """
    def __init__(self, emb_size: int):
        """
        Initialize the Attention mechanism.

        Args:
        - emb_size (int): The size of the embeddings (and hence the size of the matrices for Q, K, V).

        """
        self.emb_size = emb_size
        # Initialize weights for Q, K, V
        self.W_q = np.random.rand(emb_size, emb_size)
        self.W_k = np.random.rand(emb_size, emb_size)
        self.W_v = np.random.rand(emb_size, emb_size)
    
    def scaled_dot_product_attention(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray):
        """
        Computes scaled dot-product attention.

        Args:
        - Q (np.ndarray): Queries matrix.
        - K (np.ndarray): Keys matrix.
        - V (np.ndarray): Values matrix.

        Returns:
        - The result of the attention mechanism, and the attention weights.
        """
        d_k = Q.shape[-1]
        
        # Compute the dot product for queries and keys, then scale
        scores = np.dot(Q, K.T) / np.sqrt(d_k)

        # Apply softmax to obtain the weights on the values
        weights = softmax(scores)
        
        # Multiply by V to get the final output
        output = np.dot(weights, V)

        return output, weights

    def forward(self, x: np.ndarray):
        """
        Forward pass to project inputs and compute attention.

        Args:
        - x (np.ndarray): Input tensor of shape (sequence_length, emb_size).

        Returns:
        - The output after applying attention.
        """
        # Check input dimensions
        if x.ndim != 2 or x.shape[1] != self.emb_size:
            raise ValueError(f"Input tensor must be of shape (?, {self.emb_size})")

        # Project the input tensor into Q, K, V
        Q = np.dot(x, self.W_q)
        K = np.dot(x, self.W_k)
        V = np.dot(x, self.W_v)

        # Compute attention
        output, _ = self.scaled_dot_product_attention(Q, K, V)

        return output
