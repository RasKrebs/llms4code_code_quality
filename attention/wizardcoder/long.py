import numpy as np
from typing import Tuple

class Attention:
    """A basic attention mechanism for transformer models."""

    def __init__(self, emb_size: int):
        # Initialize weights with random values
        self.Wq = np.random.randn(emb_size, emb_size)
        self.Wk = np.random.randn(emb_size, emb_size)
        self.Wv = np.random.randn(emb_size, emb_size)

    def scaled_dot_product_attention(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Performs the scaled dot product attention operation on the given queries, keys and values."""

        # Calculate the dot product of the query and key matrices
        scores = np.matmul(Q, K.transpose())
        # Scale the scores by the dimensionality of the key embeddings (as recommended in the Transformer paper)
        scaled_scores = scores / np.sqrt(K.shape[-1])
        # Apply softmax to get attention weights
        weights = self._softmax(scaled_scores, axis=-1)

        # Weighted sum of value vectors to get the context vector
        context = np.matmul(weights, V)

        return context, weights

    def _softmax(self, x: np.ndarray, axis=None) -> np.ndarray:
        """Computes softmax values for each sets of scores in x."""

        # Subtract the max value from the set to improve numerical stability
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=axis, keepdims=True)

    def forward(self, x: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Performs the attention operation."""

        # Split input into queries and keys
        Q = self.Wq @ x
        K = self.Wk @ x
        V = self.Wv @ x

        # Perform scaled dot product attention
        return self.scaled_dot_product_attention(Q, K, V)
