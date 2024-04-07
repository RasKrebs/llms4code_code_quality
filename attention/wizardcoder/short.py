import numpy as np

class Attention:
    def __init__(self, embedding_size):
        self.embedding_size = embedding_size

        # Initializing weights randomly for simplicity
        self.Wa = np.random.randn(embedding_size, 30) # Weight matrix for attention mechanism
        self.ba = np.random.randn(30, 1)                 # Bias for attention layer

    def forward(self, inputs):
        """
        inputs shape: (embedding_size, sequence_length)
        outputs shape: (sequence_length, sequence_length)
        """
        # Calculate scores via w*x + b
        scores = np.dot(self.Wa, inputs) + self.ba

        # Normalize scores with softmax to get attention weights
        alphas = self._softmax(scores)

        return alphas

    def _softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
