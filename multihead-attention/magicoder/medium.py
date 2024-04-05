import numpy as np
from typing import Optional

class MultiHeadAttention():
    """
    A Multi-head Attention module.

    This class can be used to perform multi-head attention operation on the input data.
    """

    def __init__(self, emb_size: int, num_heads: int):
        self.emb_size = emb_size
        self.num_heads = num_heads

        assert emb_size % num_heads == 0  # ensure divisibility of the dimensions

        self.Wq = np.random.randn(emb_size, emb_size)   # query weights
        self.Wk = np.random.randn(embize, emb_size)    # key weights
        self.Wv = np.random.randn(emb_size, emb_size)     # value weights

        self.d_k = int(self.emb_size / self.num_heads)  # dimension of each head

    def scaled_dot_product_attention(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        This function implements the scaled dot-product attention mechanism for multi-head self-attention

        Args:
            Q (np.ndarray): Query matrix of shape `(batch_size, seq_len, emb_size)`
            K (np.ndarray): Key matrix of shape `(batch_size, seq_len, emb_size)`
            V (np.ndarray): Value matrix of shape `(batch_size, seq_len, emb_size)`

        Returns:
            np.ndarray: Attention output of shape `(batch_size, num_heads, seq_len, d_k)`

        Note that the implementation here doesn't handle numeric overflow yet and applies softmax directly on raw scores.
        """

        # Scaled dot-product attention
        matmul_QK = np.matmul(Q, K.transpose(-2, -1))  # (batch_size, seq_len, seq_len)
        scaled_attention_logits = matmul_QK / np.sqrt(self.d_k)

        attention_weights = np.exp(scaled_attention_logits)/np.sum(np.exp(scaled_attention_logits), axis=-1, keepdims=True)  # apply softmax

        return np.matmul(attention_weights, V)

    def forward(self, Q: np.ndarray, K: Optional[np.ndarray] = None, V: Optional[np.ndarray] = None):
        """
         This function performs the multi-head attention mechanism and return output.

         Args:
            Q (np.ndarray): Query matrix of shape `(batch_size, seq_len, emb_size)`
            K (Optional[np.ndarray]): Key matrix of shape `(batch_size, seq_len, emb_size)`
            V (Optional[np.ndarray]): Value matrix of shape `(batch_size, seq_len, emb_size)`

        Returns:
           np.ndarray: Output tensor after applying multi-head attention mechanism. Shape is `(batch_size, num_heads, seq_len, d_k)`
        """

        if K is None and V is None:
            Q = Q @ self.Wq   # (batch_size, seq_len, emb_size)
            K = Q @ self.Wk    # (batch_size, seq_len, emb_size)
            V = Q @ self.Wv     # (batch_size, seq_len, emb_size)

        batch_size = Q.shape[0]  # get the first dimension size

        Q = Q.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(2, 0, 1, 3)
        K = K.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(2, 0, 1, 3)
        V = V.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(2, 0, 1, 3)

        output = self.scaled_dot_product_attention(Q, K, V)    # (num_heads, batch_size, seq_len, d_k)

        return output.transpose(1, 2, 0, 3).reshape(batch_size, -1, self.emb_size)   # reshape back to match original shape