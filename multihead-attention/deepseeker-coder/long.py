import numpy as np

class MultiHeadAttention:
    def __init__(self, emb_size: int, num_heads: int):
        assert emb_size % num_heads == 0, "Embedding size must be divisible by number of heads"

        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_size = emb_size // num_heads

        # Initialize weights for query, key, value, and output projections
        self.query_weights = np.random.randn(emb_size, emb_size)
        self.key_weights = np.random.randn(emb_size, emb_size)
        self.value_weights = np.random.randn(emb_size, emb_size)
        self.output_weights = np.random.randn(emb_size, emb_size)

    def scaled_dot_product_attention(self, query: np.ndarray, key: np.ndarray, value: np.ndarray) -> np.ndarray:
        """
        Compute the scaled dot product attention
        """
        depth = query.shape[-1]
        scores = np.matmul(query, key.transpose(-2, -1)) / np.sqrt(depth)
        attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
        output = np.matmul(attention_weights, value)
        return output

    def forward(self, query: np.ndarray, key: np.ndarray, value: np.ndarray) -> np.ndarray:
        """
        Forward pass of the MultiHeadAttention layer
        """
        # Project input tensors into query, key, and value tensors
        query = np.matmul(query, self.query_weights)
        key = np.matmul(key, self.key_weights)
        value = np.matmul(value, self.value_weights)

        # Split query, key, and value tensors into multiple heads
        query = query.reshape(-1, query.shape[1], self.num_heads, self.head_size)
        key = key.reshape(-1, key.shape[1], self.num_heads, self.head_size)
        value = value.reshape(-1, value.shape[1], self.num_heads, self.head_size)

        # Apply attention and concatenate results
        attention_output = self.scaled_dot_product_attention(query, key, value)
        attention_output = attention_output.reshape(-1, attention_output.shape[1], self.emb_size)

        # Apply output projection
        output = np.matmul(attention_output, self.output_weights)

        return output