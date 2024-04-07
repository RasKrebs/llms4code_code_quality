import numpy as np

class Attention:
    def __init__(self, embedding_size):
        self.embedding_size = embedding_size
        self.W = np.random.randn(embedding_size, embedding_size)
    
    def forward(self, x):
        # x: input sequence of shape (sequence_length, embedding_size)
        
        # Compute the dot product of input sequence with weight matrix
        attention_scores = np.dot(x, self.W)
        
        # Compute the softmax to get attention weights
        attention_weights = np.exp(attention_scores) / np.sum(np.exp(attention_scores), axis=0)
        
        # Compute the weighted sum of input sequence using attention weights
        output = np.dot(attention_weights.T, x)
        
        return output
