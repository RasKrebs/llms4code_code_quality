import numpy as np

def scaled_dot_product_attention(query, key, value):
    """
    Calculate the attention weights. 
    query, key, value must have matching leading dimensions.
    The sequence to be attended to (key) must be transposable.
    """
    matmul_qk = np.dot(query, key.T)  # matrix multiplication for the queries and keys
    depth = query.shape[-1]
    logits = matmul_qk / np.sqrt(depth)
    attention_weights = softmax(logits, axis=-1)
    output = np.dot(attention_weights, value)
    return output, attention_weights

def softmax(x, axis=None):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

class MultiHeadAttention:
    def __init__(self, embedding_size, num_heads):
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        # Ensure the embedding size is divisible by the number of heads
        assert embedding_size % num_heads == 0

        self.depth = embedding_size // num_heads
        self.wq = np.random.rand(embedding_size, embedding_size)
        self.wk = np.random.rand(embedding_size, embedding_size)
        self.wv = np.random.rand(embedding_size, embedding_size)
        self.wo = np.random.rand(embedding_size, embedding_size)

    def split_into_heads(self, x):
        """
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = x.reshape(x.shape[0], x.shape[1], self.num_heads, self.depth)
        return np.transpose(x, perm=[0, 2, 1, 3])

    def forward(self, q, k, v):
        batch_size = q.shape[0]

        # Linear projections
        q = self.split_into_heads(np.dot(q, self.wq))  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_into_heads(np.dot(k, self.wk))  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_into_heads(np.dot(v, self.wv))  # (batch_size, num_heads, seq_len_v, depth)

        # Scaled dot-product attention for each head
        scaled_attention, _ = scaled_dot_product_attention(q, k, v)  # (batch_size, num_heads, seq_len_q, depth)
        scaled_attention = np.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        # Concatenation of heads
        concat_attention = scaled_attention.reshape(batch_size, -1, self.embedding_size)

        # Final linear layer
        output = np.dot(concat_attention, self.wo)  # (batch_size, seq_len_q, embedding_size)

        return output