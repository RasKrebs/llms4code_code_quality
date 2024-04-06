import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

class MultiHeadAttention:
    def __init__(self, emb_size, num_heads):
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_size = emb_size // num_heads

        # Initialize weights
        self.W_q = np.random.randn(emb_size, emb_size)
        self.W_k = np.random.randn(emb_size, emb_size)
        self.W_v = np.random.randn(emb_size, emb_size)
        self.W_o = np.random.randn(emb_size, emb_size)

    def split_heads(self, x, batch_size):
        x = np.reshape(x, (batch_size, -1, self.num_heads, self.head_size))
        return np.transpose(x, (0, 2, 1, 3))

    def forward(self, x):
        batch_size = x.shape[0]

        # Linear projections
        q = np.dot(x, self.W_q)
        k = np.dot(x, self.W_k)
        v = np.dot(x, self.W_v)

        # Split into heads
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Scaled dot-product attention
        attn_weights = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(self.head_size)
        attn_weights = softmax(attn_weights)

        # Apply attention
        attn_output = np.matmul(attn_weights, v)

        # Concatenate heads
        attn_output = np.transpose(attn_output, (0, 2, 1, 3))
        attn_output = np.reshape(attn_output, (batch_size, -1, self.emb_size))

        # Final linear layer
        output = np.dot(attn_output, self.W_o)

        return output
