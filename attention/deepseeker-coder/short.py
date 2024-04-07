import numpy as np

class Attention:
    def __init__(self, embedding_size):
        self.embedding_size = embedding_size
        self.weights = np.random.randn(embedding_size, embedding_size)

    def forward(self, inputs):
        # Compute the similarity between each input vector and the query vector
        similarities = np.dot(inputs.T, self.weights)

        # Compute the weights for the weighted sum
        weights = self.softmax(similarities)

        # Compute the weighted sum
        output = np.dot(inputs, weights.T)

        return output

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
