import numpy as np
from typing import List

class MultiHeadAttention:
    """Implementation of a Multi-head Attention Mechanism."""

    def __init__(self, emb_size: int, num_heads: int):
        self.emb_size = emb_size
        self.num_heads = num_heads

        # Calculate the dimension of each head
        self.dim_per_head = emb_size // num_heads
        assert self.dim_per_head * num_heads == emb_size, "emb_size must be divisible by num_heads"

        # Initialize random weight matrices for query, key, value and output projections
        self.query = np.random.randn(self.num_heads, emb_size)
        self.key = np.random.randn(self.num_heads, emb_size)
        self.value = np.random.randn(self.num_heads, emb_size)
        self.output = np.random.randn(emb_size, num_heads)

    def scaled_dot_product_attention(self, query: List[float], key: List[float], value: List[float]) -> float:
        """Perform the Scaled Dot-Product Attention operation."""
        scores = np.dot(query, key) / np.sqrt(len(key))  # Scale dot product by square root of dimension

        # Prevent numerical overflow during softmax computation
        max_value = np.max(scores)
        exp_scores = np.exp(scores - max_value)
        softmaxed_scores = exp_scores / np.sum(exp_scores)  # Apply Softmax

        weighted_values = np.dot(softmaxed_scores, value)

        return weighted_values

    def forward(self, inputs: List[float]) -> float:
        """Perform the Multi-head Attention operation."""
        # Linear transformations for queries, keys and values
        Q = np.dot(inputs, self.query)
        K = np.dot(inputs, self.key)
        V = np.dot(inputs, self.value)

        attn_outputs = []

        # Perform Scaled Dot-Product Attention for each head and concatenate results
        for q, k, v in zip(Q, K, V):
            attn_output = self.scaled_dot_product_attention(q, k, v)
            attn_outputs.append(attn_output)

        # Concatenate the attention outputs from all heads and apply output projection
        concat_output = np.concatenate(attn_outputs)

        return np.dot(concat_output, self.output)
