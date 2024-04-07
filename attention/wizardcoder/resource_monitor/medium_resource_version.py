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

class ScaledDotProductAttention:
    """Scaled dot product attention"""
    def __init__(self, emb_size: int) -> None:
        self.emb_size = emb_size

        # initialize weight matrices for query, key and value
        self.query_weights = np.random.randn(emb_size, emb_size)
        self.key_weights = np.random.randn(emb_size, emb_size)
        self.value_weights = np.random.randn(emb_size, emb_size)

    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax values for each scores in vector x."""
        e_x = np.exp(x - np.max(x))  # subtract max value to prevent overflow
        return e_x / e_x.sum()  # return probabilities

    def forward(self, embedding: Tuple[int, int]) -> np.ndarray:
        """Compute scaled dot product attention"""

        query = np.matmul(embedding.T, self.query_weights)
        key = np.matmul(embedding.T, self.key_weights)
        value = np.matmul(embedding.T, self.value_weights)

        # compute attention scores
        scores = np.dot(query, key.T) / np.sqrt(self.emb_size)

        # apply softmax to get attention probabilities
        probs = self.softmax(scores)

        # compute weighted sum of values
        output = np.matmul(probs, value)

        return output

def execute(emb_size, num_heads):
    # Load the data
    data = np.random.rand(emb_size, seq_length)
    
    # Initialize the MultiHeadAttention model
    model = ScaledDotProductAttention(emb_size)
    
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

