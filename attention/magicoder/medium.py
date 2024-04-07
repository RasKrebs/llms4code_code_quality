import numpy as np
from typing import Tuple

class Attention():
    def __init__(self, emb_size):
        self.emb_size = emb_size  # embedding size

        # generate random weight matrices for query, key, value, and output projections
        self.Wq = np.random.randn(emb_size, emb_size)
        self.Wk = np.random.randn(emb_size, emb_size)
        self.Wv = np.random.randn(emb_size, emb_size)

    def scaled_dot_product_attention(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> Tuple[np.ndarray, float]:
        # Scaled dot product attention
        matmul_qk = np.matmul(Q, K.T)  # shape (seq_len_q, seq_len_v)

        dk = self.emb_size**0.5  # scaling by sqrt(dk)
        scaled_attention_logits = matmul_qk / dk  # softmax is applied internally

        attention_weights = np.exp(scaled_attention_logits)/np.sum(np.exp(scaled_attention_logits), axis=1)

        output = np.matmul(attention_weights, V )  # shape (seq_len_q, emb_size)

        return output, np.maximum(0., scaled_attention_logits).sum()  # for preventing numerical overflow

    def forward(self, input: np.ndarray):

        Q = np.matmul(input, self.Wq)
        K = np.matmul(input, self.Wk)
        V = np.matmul(input, self.Wv)

        output, _ = self.scaled_dot_product_attention(Q, K, V)

        return output  # shape (sequence_length, emb_size)
