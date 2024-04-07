import numpy as np

class Attention:
    def __init__(self, embedding_size):
        self.embedding_size = embedding_size
        self.weights = np.random.rand(embedding_size, embedding_size)

    def forward(self, input):
        # Ensure input is two-dimensional
        assert len(input.shape) == 2
        assert input.shape[1] == self.embedding_size

        # Compute attention scores
        scores = np.dot(input, self.weights)

        # Apply softmax to get attention distribution
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        # Compute weighted sum of input vectors
        output = np.sum(input * attention, axis=0)

        return output
