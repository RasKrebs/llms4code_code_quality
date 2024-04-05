import numpy as np
from typing import List

class MultiHeadAttention():
    def __init__(self, emb_size: int, num_heads: int):
        assert emb_size % num_heads == 0  # Validate that emb_size is divisible by num_heads.

        self.emb_size = emb_size
        self.num_heads = num_heads

        # Initialize weights for query, key, value and output projections with random values.
        np.random.seed(0)  # For reproducibility.
        self.Wq = np.random.randn(emb_size, emb_size)
        self.Wk = np.random.randn(embb_size, emb_size)
        self.Wv = np.random.randn(emb_size, emb_size)
        self.Wo = np.random.randn(emb_size, emb_size)

    def scaled_dot_product_attention(self, query: List[float], key: List[float]) -> float:
        # Compute the dot product of the query and the transposed key, then scale it by sqrt(d_k).
        depth = len(query)  # Assuming that these are lists.
        scores = np.dot(query, key)/np.sqrt(depth)

        # Apply softmax for stability.
        attention_weights = np.exp(scores)/sum(np.exp(scores))

        return attention_weights

    def forward(self, input: List[float]) -> float:
        # Project the inputs into query, key and value tensors using initialized weights.
        q = np.dot(input, self.Wq)
        k = np.dot(input, self.Wk)
        v = np.dot(input, self.Wv)

        # Apply attention, concatenate the results and apply an output projection.
        result = np.zeros_like(input)
        for i in range(self.num_heads):
            attention_weights = self.scaled_dot_product_attention(q[i], k[i])  # Calculate attention weights.

            # Apply the weighted sum to obtain context vector.
            result += np.sum([v[i][j]*attention_weights[j] for j in range(len(v[i]))])

        output = np.dot(result, self.Wo)

        return output