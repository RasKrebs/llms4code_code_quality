import numpy as np
import math
from typing import Tuple

class MultiHeadAttention:
    def __init__(self, emb_size: int, num_heads: int):
        """
        Initialize the MultiHeadAttention module.

        Parameters:
        emb_size (int): The size of the embedding.
        num_heads (int): The number of attention heads.
        """
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_size = emb_size // num_heads

        # Randomly initialize weight matrices for query, key, value, and output projections
        self.query_weights = np.random.randn(emb_size, emb_size)
        self.key_weights = np.random.randn(emb_size, emb_size)
        self.value_weights = np.random.randn(emb_size, emb_size)
        self.output_weights = np.random.randn(emb_size, emb_size)

    def scaled_dot_product_attention(self, query: np.ndarray, key: np.ndarray, value: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the attention weights and the output of the scaled dot-product attention.

        Parameters:
        query (np.ndarray): The query matrix.
        key (np.ndarray): The key matrix.
        value (np.ndarray): The value matrix.

        Returns:
        Tuple[np.ndarray, np.ndarray]: The attention weights and the output of the scaled dot-product attention.
        """
        # Calculate the dot product of the query and key matrices
        dot_product = np.dot(query, key.T)

        # Scale the dot product by the square root of the embedding size
        scale = math.sqrt(self.emb_size)
        dot_product /= scale

        # Apply softmax to get the attention weights
        attention_weights = np.exp(dot_product) / np.sum(np.exp(dot_product), axis=-1, keepdims=True)

        # Calculate the output of the scaled dot-product attention
        output = np.dot(attention_weights, value)

        return attention_weights, output

    def forward(self, query: np.ndarray, key: np.ndarray, value: np.ndarray) -> np.ndarray:
        """
        Perform the multi-head attention and return the output.

        Parameters:
        query (np.ndarray): The query matrix.
        key (np.ndarray): The key matrix.
        value (np.ndarray): The value matrix.

        Returns:
        np.ndarray: The output of the multi-head attention.
        """
        # Apply the query, key, and value projections
        query = np.dot(query, self.query_weights)
        key = np.dot(key, self.key_weights)
        value = np.dot(value, self.value_weights)

        # Split the query, key, and value matrices into multiple heads
        query = query.reshape(-1, self.num_heads, self.head_size)
        key = key.reshape(-1, self.num_heads, self.head_size)
        value = value.reshape(-1, self.num_heads, self.head_size)

        # Perform the scaled dot-product attention for each head
        attention_weights, output = self.scaled_dot_product_attention(query, key, value)

        # Concatenate the output of each head
        output = output.reshape(-1, self.emb_size)

        # Apply the output projection
        output = np.dot(output, self.output_weights)

        return output