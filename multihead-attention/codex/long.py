import math
import random

class MultiHeadAttention:
    def __init__(self, emb_size: int, num_heads: int):
        """
        Initialize the MultiHeadAttention class.
        """
        if emb_size % num_heads != 0:
            raise ValueError("emb_size must be divisible by num_heads")

        self.emb_size = emb_size
        self.num_heads = num_heads

        # Initialize weights with random values
        self.query_weights = [[random.random() for _ in range(emb_size)] for _ in range(emb_size)]
        self.key_weights = [[random.random() for _ in range(emb_size)] for _ in range(emb_size)]
        self.value_weights = [[random.random() for _ in range(emb_size)] for _ in range(emb_size)]
        self.output_weights = [[random.random() for _ in range(emb_size)] for _ in range(emb_size)]

    def scaled_dot_product_attention(self, query: list, key: list, value: list):
        """
        Perform the scaled dot product attention operation.
        """
        # Compute dot product of query and key
        dot_product = sum(q * k for q, k in zip(query, key))

        # Scale by square root of query depth
        scaled_dot_product = dot_product / math.sqrt(self.emb_size)

        # Apply softmax for stability
        softmax = math.exp(scaled_dot_product) / sum(math.exp(x) for x in key)

        # Multiply by value
        result = softmax * value

        return result

    def forward(self, input_tensor: list):
        """
        Forward pass of the MultiHeadAttention.
        """
        # Project input tensor into query, key, and value tensors
        query = [sum(i * w for i, w in zip(input_tensor, row)) for row in self.query_weights]
        key = [sum(i * w for i, w in zip(input_tensor, row)) for row in self.key_weights]
        value = [sum(i * w for i, w in zip(input_tensor, row)) for row in self.value_weights]

        # Apply attention
        attention = self.scaled_dot_product_attention(query, key, value)

        # Concatenate results
        concatenated = attention + input_tensor

        # Apply output projection
        output = [sum(i * w for i, w in zip(concatenated, row)) for row in self.output_weights]

        return output
