import numpy as np

class AttentionHead:
    def __init__(self, embedding_size):
        self.Wq = np.random.randn(embedding_size, embedding_size)  # query weight
        self.Wk = np.random.randn(embedding_size, embedding_size)  # key weight
        self.Wv = np.random.randn(embedding_size, embedding_size)  # value weight

    def __call__(self, query, key):
        Q = np.matmul(query, self.Wq)
        K = np.matmul(key, self.Wk)
        V = np.matmul(key, self.Wv)  # value is also the output of attention mechanism

        scores = np.matmul(Q, K.T) / np.sqrt(embedding_size)  # scaled dot-product attention
        weights = np.exp(scores - np.max(scores)) / np.sum(np.exp(scores - np.max(scores)))  # softmax operation

        return np.matmul(weights, V)

class MultiHeadAttention:
    def __init__(self, embedding_size, num_heads):
        self.embedding_size = embedding_size
        self.num_heads = num_heads

        self.attention_heads = [AttentionHead(embedding_size) for _ in range(num_heads)]

    def __call__(self, query, key):
        return np.concatenate([ah(query, key) for ah in self.attention_heads])  # concatenating the results of each attention head