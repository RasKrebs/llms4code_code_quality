import numpy as np
from typing import Tuple

class Attention:
    def __init__(self, emb_size: int):
        """Initialize Attention with random weight matrices."""

        self.emb_size = emb_size

        # Initialize random weights for query, key, value and output projection
        self.Wq = np.random.rand(emb_size, emb_size)
        self.Wk = np.random.rand(emb_size, emb_size)
        self.Wv = np.random.rand(emb_size, emb_size)

        # Initialize output projection weights
        self.Wo = np.random.rand(emb_size, emb_size)

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """Softmax function to normalize the input array."""

        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    @staticmethod
    def scaled_dot_product_attention(query: np.ndarray, key: np.ndarray, value: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Scaled Dot Product Attention mechanism."""

        # Compute dot product between query and keys
        dot_product = np.dot(query, key.T)

        # Scale by square root of emb_size (to avoid very small numbers when doing softmax)
        scaled_dot_product = dot_product / np.sqrt(len(key))

        # Apply softmax to get attention weights
        attention_weights = Attention.softmax(scaled_dot_product)

        # Compute weighted sum of values using attention weights
        weighted_value = np.dot(attention_weights, value)

        return weighted_value, attention_weights

    def forward(self, inputs: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
        """Forward pass of Attention mechanism."""

        # Separate input into query, key and value
        query, key, value = inputs

        # Project inputs to query, key and value spaces using learned weights
        Q = np.dot(query, self.Wq)
        K = np.dot(key, self.Wk)
        V = np.dot(value, self.Wv)

        # Compute attention weighted value and attention weights
        weighted_value, _ = Attention.scaled_dot_product_attention(Q, K, V)

        # Project the weighted value to output space using learned weight matrix
        output = np.dot(weighted_value, self.Wo)

        return output
