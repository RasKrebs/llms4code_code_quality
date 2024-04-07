import numpy as np
from typing import Tuple

class Attention:
    def __init__(self, emb_size: int):
        self.emb_size = emb_size
        # Initialize weights with random values
        self.weights = np.random.rand(emb_size*3)

    @staticmethod
    def scaled_dot_product_attention(query: np.ndarray, key: np.ndarray, value: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Performs Scaled Dot Product Attention between query and keys
        :param query: numpy array of shape (emb_size,)
        :param key: numpy array of shape (emb_size,)
        :param value: numpy array of shape (emb_size,)

        :return: Tuple containing the attention weighted value and the attention weights
        """
        # Compute dot product between query and keys
        dot_product = np.dot(query, key)

        # Scale by square root of emb_size (to avoid very small numbers when doing softmax)
        scaled_dot_product = dot_product / np.sqrt(len(key))

        # Apply softmax to get attention weights
        attention_weights = np.exp(scaled_dot_product - np.max(scaled_dot_product)) / \
                              np.sum(np.exp(scaled_dot_product - np.max(scaled_dot_product)))

        # Compute weighted sum of values using attention weights
        weighted_value = np.dot(attention_weights, value)

        return weighted_value, attention_weights

    def forward(self, input_tensor: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Tuple[np.ndarray, float]:
        """
        Forward pass of Attention mechanism

        :param input_tensor: A tuple containing the query, key and value tensors
        Each tensor is a numpy array of shape (emb_size,)

        :return: Tuple containing the attention weighted value and the attention weights
        """
        # Separate the query, key and value from input_tensor
        query, key, value = input_tensor

        # Perform Scaled Dot Product Attention
        weighted_value, attention_weights = self.scaled_dot_product_attention(query, key, value)

        return weighted_value, attention_weights
