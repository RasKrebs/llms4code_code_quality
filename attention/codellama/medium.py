import numpy as np

class Attention:
    def __init__(self, embedding_size):
        # Initializing weight matrices for query, key, value and output projections
        self.wq = np.random.randn(embedding_size, embedding_size)
        self.wk = np.random.randn(embedding_size, embedding_size)
        self.wv = np.random.randn(embedding_size, embedding_size)
        self.wo = np.random.randn(embedding_size, embedding_size)

    def scaled_dot_product_attention(self, query, keys):
        # Calculating the scaled dot-product attention between query and key vectors
        scores = np.matmul(query, keys) / np.sqrt(keys.shape[1])

        # Applying softmax function to calculate weights for each value vector
        weights = self._softmax(scores)

        # Calculating the context vector by taking weighted sum of value vectors
        context_vector = np.matmul(weights, keys)

        return context_vector, weights

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))  # For numeric stability
        return e_x / e_x.sum()

    def forward(self, sequence):
        if len(sequence) == 0:
            raise ValueError("Sequence cannot be empty.")

        # Projecting the input sequence using query, key and value weight matrices
        q = np.matmul(sequence, self.wq)
        k = np.matmul(sequence, self.wk)
        v = np.matmul(sequence, self.wv)

        # Calculating the scaled dot-product attention and passing it through output projection layer
        context_vector, attention_weights = self.scaled_dot_product_attention(q, k)
        output = np.matmul(context_vector, self.wo)

        return output, attention_weights
