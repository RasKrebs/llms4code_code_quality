import numpy as np
from typing import Tuple

class Attention:
    def __init__(self, emb_size: int):
        """
        Initialize the Attention mechanism.

        Parameters:
        emb_size (int): The size of the embedding vectors.
        """
        self.emb_size = emb_size
        # Initialize weight matrices for query, key, value, and output projections
        self.W_q = np.random.randn(emb_size, emb_size)
        self.W_k = np.random.randn(emb_size, emb_size)
        self.W_v = np.random.randn(emb_size, emb_size)
        self.W_o = np.random.randn(emb_size, emb_size)

    def scaled_dot_product_attention(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the scaled dot-product attention.

        Parameters:
        Q (np.ndarray): Query matrix of shape (emb_size, seq_length)
        K (np.ndarray): Key matrix of shape (emb_size, seq_length)
        V (np.ndarray): Value matrix of shape (emb_size, seq_length)

        Returns:
        output (np.ndarray): Output matrix of shape (emb_size, seq_length)
        attn_weights (np.ndarray): Attention weights matrix of shape (seq_length, seq_length)
        """
        # Compute the dot product of Q and K^T
        dot_product = np.dot(Q, K.T)

        # Scale the dot product by the square root of the embedding size
        scale = np.sqrt(self.emb_size)
        scaled_dot_product = dot_product / scale

        # Apply softmax to get the attention weights
        attn_weights = self.softmax(scaled_dot_product)

        # Compute the output as the weighted sum of the value vectors
        output = np.dot(attn_weights, V)

        return output, attn_weights

    def softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the softmax of a matrix.

        Parameters:
        x (np.ndarray): Input matrix

        Returns:
        softmax_x (np.ndarray): Softmax matrix
        """
        # Subtract the max of x for numerical stability
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform the forward pass of the attention mechanism.

        Parameters:
        x (np.ndarray): Input matrix of shape (emb_size, seq_length)

        Returns:
        output (np.ndarray): Output matrix of shape (emb_size, seq_length)
        attn_weights (np.ndarray): Attention weights matrix of shape (seq_length, seq_length)
        """
        # Compute the query, key, and value matrices
        Q = np.dot(self.W_q, x)
        K = np.dot(self.W_k, x)
        V = np.dot(self.W_v, x)

        # Compute the output and attention weights
        output, attn_weights = self.scaled_dot_product_attention(Q, K, V)

        # Apply the output projection
        output = np.dot(self.W_o, output)

        return output, attn_weights
