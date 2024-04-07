import numpy as np

class Attention:
    """
    Class for implementing scaled dot-product attention from scratch for transformer models.
    """

    def __init__(self, emb_size: int):

        """
        Initializes the Attention layer.

        Args:
            emb_size: Embedding size of the input tensors.
        """
        super().__init__()
        self.emb_size = emb_size

        # Initialize weights with random values using a Gaussian distribution
        self.query_weights = np.random.randn(emb_size, emb_size)
        self.key_weights = np.random.randn(emb_size, emb_size)
        self.value_weights = np.random.randn(emb_size, emb_size)

        # Handle numerical instability for softmax
        self.eps = 1e-8

    def scaled_dot_product_attention(self, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Performs scaled dot-product attention.

        Args:
            q: Query tensor of shape (sequence_length, emb_size).
            k: Key tensor of shape (sequence_length, emb_size).
            v: Value tensor of shape (sequence_length, emb_size).

        Returns:
            The computed attention output tensor of shape (sequence_length, emb_size).
        """

        # Calculate attention scores (dot product of queries and keys scaled by square root of embedding size)
        scores = np.matmul(q, k.T) / np.sqrt(self.emb_size)  # (sequence_length, sequence_length)

        # Apply softmax to attention scores for normalization
        attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True)) + self.eps  # Numerical stability
        attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)

        # Calculate attention output by multiplying attention weights with values
        output = np.matmul(attn_weights, v)  # (sequence_length, emb_size)

        return output

    def forward(self, x: np.ndarray, sequence_length: int) -> np.ndarray:
        """
        Projects the input tensor into query, key, and value tensors using learned weights,
        and performs the scaled dot-product attention.

        Args:
            x: Input tensor of shape (sequence_length, emb_size).
            sequence_length: Length of the input sequence.

        Returns:
            The output tensor after attention of shape (sequence_length, emb_size).
        """

        # Project input into query, key, and value tensors
        q = np.matmul(x, self.query_weights)  # (sequence_length, emb_size)
        k = np.matmul(x, self.key_weights)    # (sequence_length, emb_size)
        v = np.matmul(x, self.value_weights)  # (sequence_length, emb_size)

        # Perform scaled dot-product attention
        output = self.scaled_dot_product_attention(q, k, v)

        return output
