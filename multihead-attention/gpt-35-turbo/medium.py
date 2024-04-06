import numpy as np

class MultiHeadAttention:
    def __init__(self, emb_size: int, num_heads: int):
        assert emb_size % num_heads == 0, "emb_size must be divisible by num_heads"
        
        self.emb_size = emb_size
        self.num_heads = num_heads
        
        self.head_size = emb_size // num_heads
        
        # Initialize weight matrices for query, key, value, and output projections
        self.W_q = np.random.randn(emb_size, emb_size)
        self.W_k = np.random.randn(emb_size, emb_size)
        self.W_v = np.random.randn(emb_size, emb_size)
        self.W_o = np.random.randn(emb_size, emb_size)
        
    def scaled_dot_product_attention(self, Q, K, V):
        d_k = K.shape[-1]
        scaled_dot_product = np.matmul(Q, K.T) / np.sqrt(d_k)
        attention_weights = np.exp(scaled_dot_product - np.max(scaled_dot_product, axis=-1, keepdims=True))
        attention_weights /= np.sum(attention_weights, axis=-1, keepdims=True)
        output = np.matmul(attention_weights, V)
        return output
        
    def forward(self, X):
        batch_size, seq_length, emb_size = X.shape
        
        # Project input X into query, key, and value matrices
        Q = np.dot(X, self.W_q)
        K = np.dot(X, self.W_k)
        V = np.dot(X, self.W_v)
        
        # Split the matrices into multiple heads
        Q = np.reshape(Q, (batch_size, seq_length, self.num_heads, self.head_size))
        K = np.reshape(K, (batch_size, seq_length, self.num_heads, self.head_size))
        V = np.reshape(V, (batch_size, seq_length, self.num_heads, self.head_size))
        
        # Transpose the dimensions to perform attention over the sequence length
        Q = np.transpose(Q, (0, 2, 1, 3))
        K = np.transpose(K, (0, 2, 1, 3))
        V = np.transpose(V, (0, 2, 1, 3))
        
        # Apply scaled dot product attention for each head
        outputs = self.scaled_dot_product_attention(Q, K, V)
        
        # Reshape and concatenate the output heads
        outputs = np.transpose(outputs, (0, 2, 1, 3))
        outputs = np.reshape(outputs, (batch_size, seq_length, emb_size))
        
        # Project the concatenated output back to the original size
        outputs = np.dot(outputs, self.W_o)
        
        return outputs
