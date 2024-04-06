import numpy as np

class MultiHeadAttention:
    def __init__(self, embedding_size, num_heads):
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.head_size = embedding_size // num_heads
        
        # Initialize random weights for query, key, value, and output projection
        self.Wq = np.random.randn(embedding_size, embedding_size)
        self.Wk = np.random.randn(embedding_size, embedding_size)
        self.Wv = np.random.randn(embedding_size, embedding_size)
        
    def forward(self, X):
        # Split input into multiple heads
        Xq = np.dot(X, self.Wq).reshape(-1, self.num_heads, self.head_size)
        Xk = np.dot(X, self.Wk).reshape(-1, self.num_heads, self.head_size)
        Xv = np.dot(X, self.Wv).reshape(-1, self.num_heads, self.head_size)
        
        # Compute attention weights
        attention_scores = np.matmul(Xq, Xk.transpose((0, 2, 1))) / np.sqrt(self.head_size)
        attention_weights = np.softmax(attention_scores, axis=-1)
        
        # Apply attention weights to values
        attention_output = np.matmul(attention_weights, Xv)
        
        # Reshape and concatenate output heads
        output = attention_output.reshape(-1, self.embedding_size)
        
        return output
