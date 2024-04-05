import numpy as np

class MultiHeadAttention:
    def __init__(self, emb_size: int, num_heads: int):
        assert emb_size % num_heads == 0, "emb_size must be divisible by num_heads"
        
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_size = emb_size // num_heads
        
        # Initialize weights for query, key, value, and output projections
        self.W_q = np.random.randn(emb_size, emb_size)
        self.W_k = np.random.randn(emb_size, emb_size)
        self.W_v = np.random.randn(emb_size, emb_size)
        self.W_o = np.random.randn(emb_size, emb_size)
        
    def scaled_dot_product_attention(self, Q, K, V):
        # Calculate scaled dot product attention
        dot_product = np.dot(Q, K.T) / np.sqrt(self.head_size)
        attention_weights = np.softmax(dot_product, axis=-1)
        output = np.dot(attention_weights, V)
        
        return output
        
    def forward(self, input_tensor):
        # Check if input shape is valid
        assert input_tensor.shape[1] == self.emb_size, "Input shape does not match embedding size"
        
        # Project input tensor into query, key, and value tensors
        Q = np.dot(input_tensor, self.W_q)
        K = np.dot(input_tensor, self.W_k)
        V = np.dot(input_tensor, self.W_v)
        
        # Split into multiple heads
        Q_split = np.array_split(Q, self.num_heads, axis=-1)
        K_split = np.array_split(K, self.num_heads, axis=-1)
        V_split = np.array_split(V, self.num_heads, axis=-1)
        
        # Apply attention for each head
        outputs = [self.scaled_dot_product_attention(Q_split[i], K_split[i], V_split[i]) for i in range(self.num_heads)]
        
        # Concatenate outputs and apply output projection
        concatenated_output = np.concatenate(outputs, axis=-1)
        output = np.dot(concatenated_output, self.W_o)
        
        return output