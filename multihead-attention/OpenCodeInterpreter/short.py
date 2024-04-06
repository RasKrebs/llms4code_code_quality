import numpy as np
from typing import List

class MultiHeadAttention:
    def __init__(self, embedding_size: int, n_heads: int):
        self.embedding_size = embedding_size
        self.n_heads = n_heads

        # Create weight matrices for each head (query, key and value)
        self.weights_q = [np.random.rand(embedding_size, embedding_size//n_heads) for _ in range(n_heads)]
        self.weights_k = [np.random.rand(embedding_size, embedding_size//n_heads) for _ in range(n_heads)]
        self.weights_v = [np.random.rand(embedding_size, embedding_size//n_heads) for _ in range(n_heads)]

    def forward(self, input_tensor: List[List[int]]) -> List[List[float]]:
        output_tensors = []

        # Perform attention for each head
        for i in range(self.n_heads):
            # Apply linear transformations to the input tensor
            q = np.matmul(input_tensor, self.weights_q[i])  # Query
            k = np.matmul(input_tensor, self.weights_k[i])  # Key
            v = np.matmul(input_tensor, self.weights_v[i])  # Value

            # Compute attention scores
            attn_scores = np.matmul(q, k.T) / np.sqrt(self.embedding_size//n_heads)

            # Apply softmax to attention scores
            attn_probs = np.exp(attn_scores - np.max(attn_scores))
            attn_probs /= np.sum(attn_probs, axis=1, keepdims=True)

            # Apply attention weights to values
            weighted_values = np.matmul(attn_probs, v)

            output_tensors.append(weighted_values)

        return output_tensors
