import numpy as np

class Attention:
    def __init__(self, emb_size: int):
        """
        Initialize the Attention class.

        Parameters:
        emb_size (int): The size of the input embeddings.
        """
        self.emb_size = emb_size
        # Initialize weights with random values
        self.W_q = np.random.randn(emb_size, emb_size)
        self.W_k = np.random.randn(emb_size, emb_size)
        self.W_v = np.random.randn(emb_size, emb_size)

    def scaled_dot_product_attention(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        Perform scaled dot product attention.

        Parameters:
        Q (np.ndarray): Query matrix.
        K (np.ndarray): Key matrix.
        V (np.ndarray): Value matrix.

        Returns:
        np.ndarray: The result of the attention operation.
        """
        d_k = K.shape[1]
        # Compute the dot product of Q and K, and scale by the square root of d_k
        scores = np.matmul(Q, K.T) / np.sqrt(d_k)
        # Apply softmax to get the attention weights
        attn_weights = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
        # Compute the weighted sum of the value vectors
        output = np.matmul(attn_weights, V)
        return output

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Perform the forward pass of the Attention mechanism.

        Parameters:
        x (np.ndarray): Input tensor of shape (sequence_length, embedding_size).

        Returns:
        np.ndarray: The result of the attention operation.
        """
        # Project input into query, key, and value vectors
        Q = np.matmul(x, self.W_q)
        K = np.matmul(x, self.W_k)
        V = np.matmul(x, self.W_v)
        # Perform attention operation
        output = self.scaled_dot_product_attention(Q, K, V)
        return output
