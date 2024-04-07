import numpy as np

class Attention:
    def __init__(self, embedding_size):
        self.embedding_size = embedding_size

        # Randomly initialize weights for query and value transformations
        self.query = np.random.rand(embedding_size)
        self.value = np.random.rand(embedding_size)

    def forward(self, embeddings):
        '''
        Inputs:
        - embeddings : a sequence of embeddings (sequence_length x embedding_size)

        Returns:
        - outputs : a weighted sum of embeddings (sequence_length x 1)
        '''

        # Dot product between query and each vector in the sequence
        scores = np.matmul(embeddings, self.query)

        # Apply softmax to get weights
        weights = np.exp(scores - np.max(scores)) / np.sum(np.exp(scores - np.max(scores)))

        # Multiply weights with corresponding embeddings and sum them up
        outputs = np.matmul(weights, embeddings)

        return outputs
