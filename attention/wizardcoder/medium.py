import numpy as np
from typing import Tuple

class ScaledDotProductAttention:
    """Scaled dot product attention"""
    def __init__(self, emb_size: int) -> None:
        self.emb_size = emb_size

        # initialize weight matrices for query, key and value
        self.query_weights = np.random.randn(emb_size, emb_size)
        self.key_weights = np.random.randn(emb_size, emb_size)
        self.value_weights = np.random.randn(emb_size, emb_size)

    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax values for each scores in vector x."""
        e_x = np.exp(x - np.max(x))  # subtract max value to prevent overflow
        return e_x / e_x.sum()  # return probabilities

    def forward(self, embedding: Tuple[int, int]) -> np.ndarray:
        """Compute scaled dot product attention"""

        query = np.matmul(embedding, self.query_weights)
        key = np.matmul(embedding, self.key_weights)
        value = np.matmul(embedding, self.value_weights)

        # compute attention scores
        scores = np.dot(query, key.T) / np.sqrt(self.emb_size)

        # apply softmax to get attention probabilities
        probs = self.softmax(scores)

        # compute weighted sum of values
        output = np.matmul(probs, value)

        return output
