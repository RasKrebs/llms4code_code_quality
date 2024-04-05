import numpy as np


class MultiHeadAttention:
  """
  Multi-head attention layer implementation.

  Args:
      embedding_size: Dimensionality of the input and output embedding.
      num_heads: Number of attention heads.
  """
  def __init__(self, embedding_size, num_heads):
    self.embedding_size = embedding_size
    self.num_heads = num_heads
    self.d_k = embedding_size // num_heads  # Dimension per head

    # Initialize weights randomly
    self.W_q = np.random.rand(embedding_size, embedding_size) / np.sqrt(embedding_size)
    self.W_k = np.random.rand(embedding_size, embedding_size) / np.sqrt(embedding_size)
    self.W_v = np.random.rand(embedding_size, embedding_size) / np.sqrt(embedding_size)
    self.W_o = np.random.rand(embedding_size, embedding_size) / np.sqrt(embedding_size)

  def forward(self, inputs):
    """
    Forward pass of the multi-head attention layer.

    Args:
        inputs: Input tensor of shape (batch_size, seq_len, embedding_size).

    Returns:
        Output tensor after applying multi-head attention of shape (batch_size, seq_len, embedding_size).
    """
    batch_size, seq_len, embedding_size = inputs.shape

    # Project queries, keys, and values
    queries = np.matmul(inputs, self.W_q)
    keys = np.matmul(inputs, self.W_k)
    values = np.matmul(inputs, self.W_v)

    # Reshape and transpose for multi-head attention
    queries = self.reshape_tensor(queries, self.num_heads, transpose=True)
    keys = self.reshape_tensor(keys, self.num_heads, transpose=True)
    values = self.reshape_tensor(values, self.num_heads, transpose=True)

    # Scaled dot-product attention
    attention_scores = np.matmul(queries, keys.transpose()) / np.sqrt(self.d_k)
    attention_scores = self.softmax(attention_scores)

    # Attention context
    context = np.matmul(attention_scores, values)

    # Concatenate and project outputs
    context = self.reshape_tensor(context, self.num_heads, transpose=False)
    output = np.matmul(context, self.W_o)

    return output

  @staticmethod
  def reshape_tensor(tensor, num_heads, transpose=False):
    """
    Reshape tensor for multi-head attention.

    Args:
        tensor: Input tensor.
        num_heads: Number of attention heads.
        transpose: Whether to transpose the tensor.

    Returns:
        Reshaped tensor.
    """
    batch_size, seq_len, dim = tensor.shape
    if transpose:
      tensor = tensor.transpose((0, 2, 1))  # (batch, dim, seq_len)
    new_shape = (batch_size, num_heads, seq_len, dim // num_heads)
    return tensor.reshape(*new_shape) if not transpose else tensor.transpose((0, 2, 1, 3))

  @staticmethod
  def softmax(x):
    """
    Softmax function implementation.
    """
    e_x = np.exp(x - x.max(axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True) 
