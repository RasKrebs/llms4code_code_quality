import numpy as np

class MultiHeadAttention:
    def __init__(self, embedding_dim, num_heads):
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.depth = embedding_dim // num_heads

        assert self.depth * num_heads == self.embedding_dim, "Embedding dimension must be divisible by number of heads"

        self.wq = np.random.rand(embedding_dim, embedding_dim)
        self.wk = np.random.rand(embedding_dim, embedding_dim)
        self.wv = np.random.rand(embedding_dim, embedding_dim)

    def split_heads(self, x):
        batch_size = x.shape[0]
        x = np.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return np.transpose(x, (0, 2, 1, 3))

    def scaled_dot_product_attention(self, q, k, v):
        matmul_qk = np.matmul(q, np.transpose(k, (0, 1, 3, 2)))
        dk = np.float32(k.shape[-1])
        scaled_attention_logits = matmul_qk / np.sqrt(dk)
        attention_weights = np.exp(scaled_attention_logits) / np.sum(np.exp(scaled_attention_logits), axis=-1, keepdims=True)
        output = np.matmul(attention_weights, v)
        return output, attention_weights

    def forward(self, v, k, q):
        batch_size = q.shape[0]

        q = np.matmul(q, self.wq)
        k = np.matmul(k, self.wk)
        v = np.matmul(v, self.wv)

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v)

        scaled_attention = np.transpose(scaled_attention, (0, 2, 1, 3))
        concat_attention = np.reshape(scaled_attention, (batch_size, -1, self.embedding_dim))

        return concat_attention