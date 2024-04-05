import numpy as np


class MultiHeadAttention:
  """
  Multi-head attention layer for transformer models.

  This class implements the multi-head attention mechanism, which allows the model to focus on different parts
  of the input at different scales.

  Args:
      emb_size (int): The size of the input embedding.
      num_heads (int): The number of attention heads.

  Raises:
      ValueError: If emb_size is not divisible by num_heads.
  """

  def __init__(self, emb_size: int, num_heads: int) -> None:
    if emb_size % num_heads != 0:
      raise ValueError("emb_size must be divisible by num_heads")

    self.d_model = emb_size
    self.num_heads = num_heads
    self.d_k = emb_size // num_heads  # Dimensionality per head

    # Initialize weights with random values
    self.w_q = np.random.randn(self.d_model, self.d_k)
    self.w_k = np.random.randn(self.d_model, self.d_k)
    self.w_v = np.random.randn(self.d_model, self.d_k)
    self.w_o = np.random.randn(self.d_k * self.num_heads, self.d_model)

  def scaled_dot_product_attention(self, query: np.ndarray, key: np.ndarray, value: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """
    Performs scaled dot-product attention.

    Args:
        query (np.ndarray): The query tensor of shape (batch_size, seq_len, d_model).
        key (np.ndarray): The key tensor of shape (batch_size, seq_len, d_model).
        value (np.ndarray): The value tensor of shape (batch_size, seq_len, d_model).
        mask (np.ndarray, optional): A mask tensor of shape (batch_size, 1, 1, seq_len) to prevent attention to padded elements. Defaults to None.

    Returns:
        np.ndarray: The attention output tensor of shape (batch_size, seq_len, d_model).
    """
    scores = np.matmul(query, key.transpose(0, 2, 1)) / np.sqrt(self.d_k)  # Scale by sqrt(d_k)

    if mask is not None:
      # Apply mask to prevent attention to padded elements
      scores += (1e-9 - np.inf) * mask

    # Apply softmax for attention weights
    attn_weights = np.softmax(scores, axis=-1)

    # Attention output (weighted sum of values)
    output = np.matmul(attn_weights, value)

    return output

  def forward(self, query: np.ndarray, key: np.ndarray, value: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """
    Performs the forward pass of the multi-head attention layer.

    Args:
        query (np.ndarray): The query tensor of shape (batch_size, seq_len, d_model).
        key (np.ndarray): The key tensor of shape (batch_size, seq_len, d_model).
        value (np.ndarray): The value tensor of shape (batch_size, seq_len, d_model).
        mask (np.ndarray, optional): A mask tensor of shape (batch_size, 1, 1, seq_len) to prevent attention to padded elements. Defaults to None.

    Returns:
        np.ndarray: The output tensor of shape (batch_size, seq_len, d_model).
    """

    # Project queries, keys, and values
    q_heads = np.matmul(query, self.w_q).reshape(query.shape[0], query.shape[1], self.num_heads, self.d_k)
    k_heads = np.matmul(key, self.w_k).reshape(key.shape[0], key.shape[1], self.num_heads, self.d_k)
    v_heads = np.matmul(value, self.w_v).reshape(value.shape[0], # value.shape[1], self.num_heads, self.d_k)
    # Apply attention using scaled_dot_product_attention
    attn_heads = [self.scaled_dot_product_attention(q, k, v, mask) for q, k, v in zip(q_heads.transpose(0, 2, 1, 3), k_heads.transpose(0, 2, 1, 3), v_heads.transpose(0, 2, 1, 3))]

    # Concatenate heads
    concat = np.concatenate(attn_heads, axis=-1)

    # Apply output projection
    output = np.matmul(concat, self.w_o)

    return output
