import psutil
import os
import threading
import multiprocessing

# Define a global variable to store the maximum resources usage
max_resources_usage = {"cpu": 0, "memory": 0}



import numpy as np

np.random.seed(42)
seq_length = 100 
emb_size = 512


# Implement the resource monitor
def resource_monitor():
    """
    Monitors the CPU and memory usage of the current process, updating global max usage.
    """
    global max_resources_usage
    process = psutil.Process(os.getpid())
    
    while monitoring:
        cpu_usage = process.cpu_percent(interval=1) / multiprocessing.cpu_count()
        memory_usage = process.memory_info().rss
        max_resources_usage['cpu'] = max(max_resources_usage['cpu'], cpu_usage)
        max_resources_usage['memory'] = max(max_resources_usage['memory'], memory_usage)



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

def execute(emb_size, num_heads):
    # Load the data
    Q = np.random.rand(seq_length, emb_size)
    K = np.random.rand(seq_length, emb_size)
    V = np.random.rand(seq_length, emb_size)
    
    # Initialize the MultiHeadAttention model
    model = Attention(emb_size)
    
    # Perform the forward pass and return the output
    output = model.forward(Q, K, V)
    return output




if __name__ == "__main__":
    # Start the resource monitoring in a separate thread
    global monitoring
    monitoring = True
    monitor_thread = threading.Thread(target=resource_monitor)
    monitor_thread.start()

    # Execute the Huffman coding process

    # Using the execute function
    output = execute(emb_size,seq_length)


    # Stop the monitoring
    monitoring = False
    monitor_thread.join()

    print(max_resources_usage['cpu']), print(max_resources_usage['memory'] / (1024**2))

