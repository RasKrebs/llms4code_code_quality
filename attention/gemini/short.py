import numpy as np

class Attention:
    """
    An attention layer for assigning weights to elements in a sequence.
    """
    def __init__(self, embedding_size, sequence_length):
        """
        Initializes the attention layer with random weights.

        Args:
          embedding_size: The size of the embedding vectors.
          sequence_length: The length of the input sequence.
        """
        self.W = np.random.rand(embedding_size, embedding_size)
        self.v = np.random.rand(1, embedding_size)
        self.sequence_length = sequence_length

    def forward(self, X):
        """
        Calculates the attention weights for a given input sequence.

        Args:
          X: A numpy array of shape (sequence_length, embedding_size) representing the input sequence.

        Returns:
          A numpy array of shape (sequence_length, 1) containing the attention weights for each element.
        """
        # Project the input sequence using the weight matrix
        u = np.tanh(X @ self.W)  # (sequence_length, embedding_size)

        # Calculate the attention scores
        scores = u @ self.v.T  # (sequence_length, 1)

        # Apply softmax to normalize the scores
        attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=0, keepdims=True)

        return attention_weights
