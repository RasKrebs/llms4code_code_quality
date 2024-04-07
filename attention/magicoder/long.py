import numpy as np
from typing import Union

class Attention(object):
    """Class representing a Scaled Dot-Product Attention layer.
    Args:
        emb_size (int): The size of the input embedding for each token.

    Returns:
        An attention object that can be called to perform attention.
    """
    def __init__(self, emb_size: int) -
        self.emb_size = emb_size

        # Initialize weights randomly for query, key and value.
        self.Wq = np.random.rand(self.emb_size, self.emb_size)
        self.Wk = np.random.rand(self.emb_size, self.emb_size)
        self.Wv = np.random.rand(self.emb_size, self.emb_size)

    def scaled_dot_product_attention(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray):
        """Compute Scaled Dot-Product Attention and return the weighted sum of values.
        Args:
            Q (numpy array): Query tensor with shape [..., seq_len_q, depth].
            K (numpy array): Key tensor with shape [..., seq_len_kv, depth].
            V (numpy array): Value tensor with shape [..., seq_len_kv, depth].
        Returns:
            output of scaled dot product attention.
        """

        # Compute the dot product between Query and Key to get the raw attention scores.
        matmul_qk = np.matmul(Q, K.T)

        # Scale the attention score by sqrt of depth (which is emb_size in your case).
        scaled_attention_logits = matmul_qk / np.sqrt(self.emb_size)

        # Compute softmax over the last axis to normalize the attention weights.
        attention_weights = np.exp(scaled_attention_logits) / np.sum(np.exp(scaled_attention_logits), axis=-1, keepdims=True)

        # Compute weighted sum of values by multiplying attention weights with Value.
        output = np.matmul(attention_weights, V)

        return output

    def forward(self, x: np.ndarray):
        """Perform the scaled dot product attention operation.
            Args:
                x (numpy array): Input tensor with shape [..., seq_len, depth].
            Returns:
                Output of the attention layer.
        """

        # Query, Key and Value tensors are obtained by applying linear transformation to input x.
        Q = np.matmul(x, self.Wq)
        K = np.matmul(x, self.Wk)
        V = np.matmul(x, self.Wv)

        # Call the scaled_dot_product_attention method to compute attention scores and weighted sum of values.
        output = self.scaled_dot_product_attention(Q, K, V)

        return output
