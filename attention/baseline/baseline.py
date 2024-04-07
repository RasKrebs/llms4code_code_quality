"""
This module contains a simple implementation of attention. The code draws
inspiration from a mixture of sources.
"""

import numpy as np


class Attention:
    """Attention layer.

    Attributes:
        emb_size (int): The size of the input embedding.
        weights_q (ndarray): The weights for the query projection.
        weights_k (ndarray): The weights for the key projection.
        weights_v (ndarray): The weights for the value projection.
        weights_o (ndarray): The weights for the output projection.
    """

    def __init__(self, emb_size:int):
        """Initializes the Attention layer.

        Args:
            emb_size (int): The size of the input embedding.

        Returns:
            None
        """
        # Save attributes
        self.emb_size = emb_size

        # Initialize random weights for query, key, value
        self.weights_q = np.random.rand(emb_size, emb_size)
        self.weights_k = np.random.rand(emb_size, emb_size)
        self.weights_v = np.random.rand(emb_size, emb_size)

        # Final linear projection weights
        self.weights_o = np.random.rand(emb_size, emb_size)

        # Softmax function
        self.softmax = lambda x: np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)), axis=0)

    def scaled_dot_product_attention(self, query: np.ndarray, key: np.ndarray, value: np.ndarray):
        """
        Compute the scaled dot-product attention.

        Args:
            query (np.ndarray): Query matrix
            key (np.ndarray): Key matrix
            value (np.ndarray): Value matrix

        Returns:
            output (np.ndarray): Output matrix
            attn_weights (np.ndarray): Attention weights matrix
        """
        # Compute the dot product of Q and K^T
        dot_product = np.dot(query, key.T)

        # Scale the dot product by the square root of the embedding size
        scale = np.sqrt(self.emb_size)
        scaled_dot_product = dot_product / scale

        # Apply softmax to get the attention weights
        attn_weights = self.softmax(scaled_dot_product)

        # Compute the output as the weighted sum of the value vectors
        output = np.dot(attn_weights, value)

        return output

    def forward(self, data: np.ndarray):
        """Performs forward pass of the MultiHeadAttention layer.

        Args:
            x (ndarray): The input tensor.

        Returns:
            ndarray: The output tensor after attention.
        """
        # Get the shape of the data input
        rows = data.shape[0]
        seq = data.shape[1]

        # Perform linear projects for query, key, value
        query = np.concatenate([np.dot(data.T, self.weights_q[i]) for i in range(self.emb_size)],
                               axis=0)
        key = np.concatenate([np.dot(data.T, self.weights_k[i]) for i in range(self.emb_size)],
                             axis=0)
        value = np.concatenate([np.dot(data.T, self.weights_v[i]) for i in range(self.emb_size)],
                               axis=0)

        # Reshape for outputs of each head
        query = query.reshape(self.emb_size, seq)
        key = key.reshape(self.emb_size, seq)
        value = value.reshape(self.emb_size, seq)

        # Scaled dot-product attention
        heads = []
        for i in range(seq):
            heads.append(self.scaled_dot_product_attention(query[:, i], key[:, i], value[:, i]))

        # Concatenate the heads
        head = np.concatenate(heads, axis=-1)

        # Final linear projection
        output = np.dot(head.reshape(rows, -1).T, self.weights_o)

        return output
