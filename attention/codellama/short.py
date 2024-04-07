import random
import numpy as np

class Attention(object):
    def __init__(self, embedding_size):
        """Initialization function for the attention layer.

        Args:
            embedding_size (int): Embedding size for the input sequence.
        """
        self.embedding_size = embedding_size

        # Initialize random weights
        self.weights = np.random.randn(self.embedding_size) / np.sqrt(self.embedding_size)

    def forward(self, x):
        """Forward pass for the attention layer.

        Args:
            x (numpy array): Input sequence of shape (sequence_length, embedding_size).

        Returns:
            context_vector (numpy array): Context vector after applying attention.
        """
        # Calculate scores and apply softmax
        scores = np.dot(self.weights, x.T)   # shape: (1, sequence_length)
        alpha = self._softmax(scores)        # shape: (1, sequence_length)

        # Compute context vector as weighted average of input vectors
        context_vector = np.sum(alpha * x, axis=0)  # shape: (embedding_size,)

        return context_vector

    def _softmax(self, x):
        """Softmax function for applying attention weights."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
