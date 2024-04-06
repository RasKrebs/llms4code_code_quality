import psutil
import os
import threading
import multiprocessing

# Define a global variable to store the maximum resources usage
max_resources_usage = {"cpu": 0, "memory": 0}


import numpy as np
import random

np.random.seed(42)
seq_length = 100 
batch_size = 32
emb_size = 512
num_heads = 8 


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


class MultiHeadAttention:
  """
  This class implements the MultiHeadAttention mechanism.

  Args:
      emb_size: The embedding size of the input.
      num_heads: The number of attention heads.
  """

  def __init__(self, emb_size: int, num_heads: int) -> None:
    if emb_size % num_heads != 0:
      raise ValueError("Embedding size must be divisible by number of heads.")

    self.d_k = emb_size // num_heads  # Dimensionality per head
    self.num_heads = num_heads

    # Initialize weight matrices with random values
    self.W_q = np.random.randn(emb_size, self.d_k) / np.sqrt(self.d_k)
    self.W_k = np.random.randn(emb_size, self.d_k) / np.sqrt(self.d_k)
    self.W_v = np.random.randn(emb_size, self.d_k) / np.sqrt(self.d_k)
    self.W_o = np.random.randn(self.d_k * self.num_heads, emb_size) / np.sqrt(emb_size)

  @staticmethod
  def scaled_dot_product_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Performs scaled dot-product attention with handling for numeric overflow.

    Args:
        q: The query matrix.
        k: The key matrix.
        v: The value matrix.
        eps: A small epsilon value for numerical stability.

    Returns:
        The attention output matrix.
    """
    scores = np.matmul(q, k.T) / np.sqrt(k.shape[1])  # Scaled dot product
    scores = np.clip(scores, -1e12, 1e12)  # Prevent overflow
    attention_weights = softmax(scores, axis=-1)
    return np.matmul(attention_weights, v)

  def forward(self, X: np.ndarray) -> np.ndarray:
    """
    Performs the multi-head attention on the input.

    Args:
        X: The input tensor.

    Returns:
        The multi-head attention output.
    """
    batch_size, seq_len, emb_size = X.shape

    # Project input to query, key, and value matrices
    Q = np.matmul(X, self.W_q)
    K = np.matmul(X, self.W_k)
    V = np.matmul(X, self.W_v)

    # Reshape and split into heads
    Q = Q.reshape(batch_size, seq_len, self.num_heads, self.d_k)
    K = K.reshape(batch_size, seq_len, self.num_heads, self.d_k)
    V = V.reshape(batch_size, seq_len, self.num_heads, self.d_k)

    # Perform attention for each head
    heads = []
    for h in range(self.num_heads):
      head_q, head_k, head_v = Q[:, :, h, :], K[:, :, h, :], V[:, :, h, :]
      head_out = self.scaled_dot_product_attention(head_q, head_k, head_v)
      heads.append(head_out)

    # Concatenate heads and project back to original dimension
    heads = np.concatenate(heads, axis=-1)
    out = np.matmul(heads, self.W_o)

    return out.reshape(batch_size, seq_len, emb_size)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
  """
  Softmax function implementation.
  """
  exp_x = np.exp(x)
  return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
def execute(batch_size, emb_size, num_heads, seq_length):
    """
    Executes the MultiHeadAttention model with generated input data.
    
    Args:
        batch_size (int): The number of samples in the batch.
        emb_size (int): The size of each input embedding vector.
        num_heads (int): The number of attention heads.
    
    Returns:
        ndarray: The output from the MultiHeadAttention model.
    """
    # Load the data
    data = np.random.rand(batch_size, seq_length, emb_size)
    
    # Initialize the MultiHeadAttention model
    model = MultiHeadAttention(emb_size=emb_size, num_heads=num_heads)
    
    # Perform the forward pass and return the output
    output = model.forward(data)
    return output



if __name__ == "__main__":
    # Start the resource monitoring in a separate thread
    global monitoring
    monitoring = True
    monitor_thread = threading.Thread(target=resource_monitor)
    monitor_thread.start()

    # Execute the Huffman coding process

    # Using the execute function
    output = execute(batch_size, emb_size, num_heads, seq_length)


    # Stop the monitoring
    monitoring = False
    monitor_thread.join()

    print(max_resources_usage['cpu']), print(max_resources_usage['memory'] / (1024**2))

