import numpy as np

class Attention:
    def __init__(self, emb_size: int):
        """
        Initialize Attention class with input embedding size.

        Args:
        emb_size (int): Size of input embeddings
        """
        self.emb_size = emb_size
        self.Wq = np.random.randn(emb_size, emb_size)
        self.Wk = np.random.randn(emb_size, emb_size)
        self.Wv = np.random.randn(emb_size, emb_size)

    def scaled_dot_product_attention(self, Q: np.array, K: np.array, V: np.array) -> np.array:
        """
        Perform scaled dot-product attention operation.

        Args:
        Q (np.array): Query tensor of shape (emb_size, sequence_length)
        K (np.array): Key tensor of shape (emb_size, sequence_length)
        V (np.array): Value tensor of shape (emb_size, sequence_length)

        Returns:
        np.array: Result of attention operation
        """
        d_k = Q.shape[0]
        att_scores = np.matmul(Q.T, K) / np.sqrt(d_k)
        att_weights = np.exp(att_scores - np.max(att_scores, axis=1, keepdims=True))
        att_weights /= np.sum(att_weights, axis=1, keepdims=True)
        att_output = np.matmul(att_weights, V.T)
        return att_output

    def forward(self, input_tensor: np.array) -> np.array:
        """
        Forward pass of the attention mechanism.

        Args:
        input_tensor (np.array): Input tensor of shape (emb_size, sequence_length)

        Returns:
        np.array: Result of attention operation
        """
        Q = np.matmul(self.Wq, input_tensor)
        K = np.matmul(self.Wk, input_tensor)
        V = np.matmul(self.Wv, input_tensor)
        return self.scaled_dot_product_attention(Q, K, V)
