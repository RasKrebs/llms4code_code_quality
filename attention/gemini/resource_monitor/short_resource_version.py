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

class Attention:
  """
  An attention layer for assigning weights to elements in a sequence.
  """
  def __init__(self, embedding_size, sequence_length):
    """
    Initializes the attention layer with random weights.

    Args:
      embedding_size: The size of the embedding vectors.
      sequence_length: The length of the input sequence.
    """
    self.W = np.random.rand(embedding_size, embedding_size)
    self.v = np.random.rand(1, embedding_size)
    self.sequence_length = sequence_length

  def forward(self, X):
    """
    Calculates the attention weights for a given input sequence.

    Args:
      X: A numpy array of shape (sequence_length, embedding_size) representing the input sequence.

    Returns:
      A numpy array of shape (sequence_length, 1) containing the attention weights for each element.
    """
    # Project the input sequence using the weight matrix
    u = np.tanh(X @ self.W)  # (sequence_length, embedding_size)

    # Calculate the attention scores
    scores = u @ self.v.T  # (sequence_length, 1)

    # Apply softmax to normalize the scores
    attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=0, keepdims=True)

    return attention_weights

def execute(emb_size, num_heads):
    # Load the data
    data = np.random.rand(seq_length, emb_size)
    
    # Initialize the MultiHeadAttention model
    model = Attention(emb_size, sequence_length=seq_length)
    
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
    output = execute(emb_size,seq_length)


    # Stop the monitoring
    monitoring = False
    monitor_thread.join()

    print(max_resources_usage['cpu']), print(max_resources_usage['memory'] / (1024**2))

