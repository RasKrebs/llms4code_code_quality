import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
  """
  Softmax function for attention weights.

  Args:
      x: A numpy array of any shape.

  Returns:
      A numpy array with the same shape as x, containing the softmax applied.
  """
  e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
  return e_x / np.sum(e_x, axis=1, keepdims=True)

class Attention:
  """
  Attention mechanism for sequence models.

  This class implements the scaled dot-product attention with random weight matrices.

  Attributes:
      d_k: Embedding size (size of query, key, and value vectors).
      W_q: Query weight matrix.
      W_k: Key weight matrix.
      W_v: Value weight matrix.
      W_o: Output projection weight matrix.
  """

  def __init__(self, embedding_size: int) -> None:
    """
    Initializes the attention layer.

    Args:
        embedding_size: Embedding size for the input and output vectors.
    """
    self.d_k = embedding_size
    self.W_q = np.random.randn(embedding_size, embedding_size) / np.sqrt(embedding_size)  # Xavier initialization
    self.W_k = np.random.randn(embedding_size, embedding_size) / np.sqrt(embedding_size)
    self.W_v = np.random.randn(embedding_size, embedding_size) / np.sqrt(embedding_size)
    self.W_o = np.random.randn(embedding_size, embedding_size) / np.sqrt(embedding_size)

  def scaled_dot_product_attention(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Scaled dot-product attention with overflow prevention.

    Args:
        Q: Query vectors (shape: [num_heads, sequence_length, d_k]).
        K: Key vectors (shape: [num_heads, sequence_length, d_k]).
        V: Value vectors (shape: [num_heads, sequence_length, d_v]).

    Returns:
        The attention output after applying softmax (shape: [num_heads, sequence_length, d_v]).
    """
    # Scaled dot product with square root d_k for variance normalization
    scores = np.matmul(Q, K.transpose()) / np.sqrt(self.d_k)
    # Prevent overflow with a large negative value for very large scores
    scores = np.clip(scores, -1e9, 1e9)
    # Apply softmax to get attention weights
    attn_weights = softmax(scores)
    # Context vector as weighted sum of values
    output = np.matmul(attn_weights, V)
    return output

  def forward(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Forward pass of the attention layer.

    Args:
        Q: Query vectors (shape: [sequence_length, d_k]).
        K: Key vectors (shape: [sequence_length, d_k]).
        V: Value vectors (shape: [sequence_length, d_v]).

    Returns:
        The attention output after projection (shape: [sequence_length, d_model]).
    """
    # Project queries, keys, and values
    Q_proj = np.matmul(Q, self.W_q)
    K_proj = np.matmul(K, self.W_k)
    V_proj = np.matmul(V, self.W_v)

    # Apply scaled dot-product attention
    attn_out = self.scaled_dot_product_attention(Q_proj, K_proj, V_proj)

    # Project the attention output
    output = np.matmul(attn_out, self.W_o)
    return output
