import numpy as np

class Attention:
    """
    Attention mechanism implementation from scratch in Python.
    """
    def __init__(self, emb_size: int, seq_len: int):
        """
        Initialize the Attention mechanism with the given embedding size and sequence length.
        Random weight matrices for query, key, value, and output projections are generated.
        """
        self.emb_size = emb_size
        self.seq_len = seq_len
        self.query_weights = np.random.rand(emb_size, emb_size)
        self.key_weights = np.random.rand(emb_size, emb_size)
        self.value_weights = np.random.rand(emb_size, emb_size)
        self.output_weights = np.random.rand(emb_size, emb_size)

    def scaled_dot_product_attention(self, query: np.ndarray, key: np.ndarray, value: np.ndarray):
        """
        Scaled dot-product attention mechanism.
        It can handle numeric overflow.
        """
        matmul_qk = np.matmul(query, np.transpose(key))

        # scale matmul_qk
        depth = key.shape[-1]
        logits = matmul_qk / np.sqrt(depth)

        # softmax is applied to axis -1 so that the scores
        # add up to 1.
        attention_weights = self.softmax(logits, axis=-1)

        output = np.matmul(attention_weights, value)

        return output, attention_weights

    def softmax(self, x: np.ndarray, axis=None):
        """
        Compute softmax values for each sets of scores in x.
        """
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / e_x.sum(axis=axis, keepdims=True)

    def forward(self, x: np.ndarray):
        """
        Forward method performs the computation and returns the output.
        """
        query = np.matmul(x, self.query_weights)
        key = np.matmul(x, self.key_weights)
        value = np.matmul(x, self.value_weights)

        out, _ = self.scaled_dot_product_attention(query, key, value)
        out = np.matmul(out, self.output_weights)

        return out
