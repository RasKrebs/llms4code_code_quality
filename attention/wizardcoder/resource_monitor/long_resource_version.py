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
from typing import Tuple

class Attention:
    """A basic attention mechanism for transformer models."""

    def __init__(self, emb_size: int):
        # Initialize weights with random values
        self.Wq = np.random.randn(emb_size, emb_size)
        self.Wk = np.random.randn(emb_size, emb_size)
        self.Wv = np.random.randn(emb_size, emb_size)

    def scaled_dot_product_attention(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Performs the scaled dot product attention operation on the given queries, keys and values."""

        # Calculate the dot product of the query and key matrices
        scores = np.matmul(Q, K.transpose())
        # Scale the scores by the dimensionality of the key embeddings (as recommended in the Transformer paper)
        scaled_scores = scores / np.sqrt(K.shape[-1])
        # Apply softmax to get attention weights
        weights = self._softmax(scaled_scores, axis=-1)

        # Weighted sum of value vectors to get the context vector
        context = np.matmul(weights, V)

        return context, weights

    def _softmax(self, x: np.ndarray, axis=None) -> np.ndarray:
        """Computes softmax values for each sets of scores in x."""

        # Subtract the max value from the set to improve numerical stability
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=axis, keepdims=True)

    def forward(self, x: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Performs the attention operation."""

        # Split input into queries and keys
        Q = self.Wq @ x
        K = self.Wk @ x
        V = self.Wv @ x

        # Perform scaled dot product attention
        return self.scaled_dot_product_attention(Q, K, V)

def execute(emb_size, num_heads):
    # Load the data
    data = np.random.rand(emb_size, seq_length)
    
    # Initialize the MultiHeadAttention model
    model = Attention(emb_size)
    
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

