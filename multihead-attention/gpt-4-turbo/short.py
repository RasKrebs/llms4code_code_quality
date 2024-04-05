import numpy as np

class MultiHeadAttention:
    def __init__(self, embedding_size, num_heads):
        assert embedding_size % num_heads == 0, "Embedding size must be divisible by number of heads."

        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.depth = embedding_size // num_heads
        
        # Initialize weights for Q, K, V, and the final linear layer
        self.W_Q = np.random.randn(embedding_size, embedding_size)
        self.W_K = np.random.randn(embedding_size, embedding_size)
        self.W_V = np.random.randn(embedding_size, embedding_size)
        self.W_O = np.random.randn(embedding_size, embedding_size)
    
    def split_heads(self, x):
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1, self.num_heads, self.depth).transpose(0, 2, 1, 3)
    
    def scaled_dot_product_attention(self, Q, K, V):
        matmul_qk = np.matmul(Q, K.transpose(0,1,3,2))
        
        # Scale the product
        dk = np.sqrt(K.shape[-1])
        scaled_attention_logits = matmul_qk / dk
        
        # Apply softmax
        attention_weights = np.exp(scaled_attention_logits) / np.sum(np.exp(scaled_attention_logits), axis=-1, keepdims=True)
        
        # Weighted sum to get output
        output = np.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        Q = np.dot(x, self.W_Q)
        K = np.dot(x, self.W_K)
        V = np.dot(x, self.W_V)
        
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        attention_output, _ = self.scaled_dot_product_attention(Q, K, V)
        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.embedding_size)
        
        output = np.dot(attention_output, self.W_O)
        return output
