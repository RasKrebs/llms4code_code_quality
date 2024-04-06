import numpy as np
from scipy.special import softmax

class MultiHeadAttention:
    def __init__(self, embed_size=512, num_heads=8):
        self.embed_size = embed_size
        self.num_heads = num_heads

        assert (self.embed_size % self.num_heads) == 0

        # Note that we don't end up backpropagating over any of the individual components, because they are constant for all inputs in one pass
        self.d_k = embed_size // num_heads  # Embedding dimension per head
        self.query = np.random.randn(embed_size, self.d_k)
        self.key = np.random.randn(embed_size, self.d_k)
        self.value = np.random.randn(embed_size, self.d_k)

    def forward(self, x):
        # Reshape the input tensor to [batch size, sequence length, embed size]
        batch_size, sequence_length, _ = x.shape
        reshaped_x = np.reshape(x, (batch_size, sequence_length, self.num_heads, self.d_k))

        # Compute Q, K, V for each head
        Q = np.dot(reshaped_x, self.query)  # [batch size, sequence length, num heads, d_k]
        K = np.dot(reshaped_x, self.key)  # [batch size, sequence length, num heads, d_k]
        V = np.dot(reshaped_x, self.value)  # [batch size, sequence length, num heads, d_k]

        # Compute scaled dot product attention for each head
        scores = np.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)  # [batch size, num heads, sequence length, sequence length]
        weights = softmax(scores, axis=-1)  # [batch size, num heads, sequence length, sequence length]

        attention = np.matmul(weights, V)  # [batch size, num heads, sequence length, d_k]
        result = np.concatenate(np.split(attention, self.num_heads, axis=2), axis=-1)  # [batch size, sequence length, embed size]

        return result
