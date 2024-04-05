import numpy as np
from scipy.special import softmax

class MultiHeadAttention:
    """
    A class to perform multi-head attention mechanism.
    """
    def __init__(self, emb_size: int, num_heads: int):
        """
        Initialize the class with the given embedding size and number of heads.

        Args:
        emb_size (int): The size of the embedding.
        num_heads (int): The number of attention heads.
        """
        assert emb_size % num_heads == 0, "Embedding size must be divisible by number of heads"

        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads

        # Initialize weight matrices for query, key, value, and output projections
        self.W_q = np.random.rand(emb_size, emb_size)
        self.W_k = np.random.rand(emb_size, emb_size)
        self.W_v = np.random.rand(emb_size, emb_size)
        self.W_o = np.random.rand(emb_size, emb_size)

    def scaled_dot_product_attention(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        Perform the scaled dot-product attention mechanism.

        Args:
        Q (np.ndarray): The query matrix.
        K (np.ndarray): The key matrix.
        V (np.ndarray): The value matrix.

        Returns:
        np.ndarray: The result of the attention mechanism.
        """
        # Calculate the dot product of Q and K, and scale by the square root of the head dimension
        scaled_dot_product = np.dot(Q, K.T) / np.sqrt(self.head_dim)

        # Apply softmax to the scaled dot product
        attention_weights = softmax(scaled_dot_product, axis=-1)

        # Multiply the attention weights by V to get the output
        output = np.dot(attention_weights, V)

        return output

    def forward(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        Perform the multi-head attention mechanism and return the output.

        Args:
        Q (np.ndarray): The query matrix.
        K (np.ndarray): The key matrix.
        V (np.ndarray): The value matrix.

        Returns:
        np.ndarray: The result of the multi-head attention mechanism.
        """
        # Project the input matrices using the weight matrices
        Q_proj = np.dot(Q, self.W_q)
        K_proj = np.dot(K, self.W_k)
        V_proj = np.dot(V, self.W_v)

        # Split the projected matrices into multiple heads
        Q_split = np.split(Q_proj, self.num_heads, axis=0)
        K_split = np.split(K_proj, self.num_heads, axis=0)
        V_split = np.split(V_proj, self.num_heads, axis=0)

        # Perform the scaled dot-product attention for each head
        outputs = [self.scaled_dot_product_attention(Q_head, K_head, V_head) for Q_head, K_head, V_head in zip(Q_split, K_split, V_split)]

        # Concatenate the outputs and project using the output weight matrix
        output_concat = np.concatenate(outputs, axis=0)
        output = np.dot(output_concat, self.W_o)

        return output
