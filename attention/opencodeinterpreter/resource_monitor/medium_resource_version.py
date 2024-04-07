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
    def __init__(self, emb_size: int):
        """Initialize Attention with random weight matrices."""

        self.emb_size = emb_size

        # Initialize random weights for query, key, value and output projection
        self.Wq = np.random.rand(emb_size, emb_size)
        self.Wk = np.random.rand(emb_size, emb_size)
        self.Wv = np.random.rand(emb_size, emb_size)

        # Initialize output projection weights
        self.Wo = np.random.rand(emb_size, emb_size)

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """Softmax function to normalize the input array."""

        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    @staticmethod
    def scaled_dot_product_attention(query: np.ndarray, key: np.ndarray, value: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Scaled Dot Product Attention mechanism."""

        # Compute dot product between query and keys
        dot_product = np.dot(query, key.T)

        # Scale by square root of emb_size (to avoid very small numbers when doing softmax)
        scaled_dot_product = dot_product / np.sqrt(len(key))

        # Apply softmax to get attention weights
        attention_weights = Attention.softmax(scaled_dot_product)

        # Compute weighted sum of values using attention weights
        weighted_value = np.dot(attention_weights, value)

        return weighted_value, attention_weights

    def forward(self, inputs: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
        """Forward pass of Attention mechanism."""

        # Separate input into query, key and value
        query, key, value = inputs

        # Project inputs to query, key and value spaces using learned weights
        Q = np.dot(query, self.Wq)
        K = np.dot(key, self.Wk)
        V = np.dot(value, self.Wv)

        # Compute attention weighted value and attention weights
        weighted_value, _ = Attention.scaled_dot_product_attention(Q, K, V)

        # Project the weighted value to output space using learned weight matrix
        output = np.dot(weighted_value, self.Wo)

        return output

def execute(emb_size, num_heads):
    # Load the data
    data = np.random.rand(seq_length, emb_size)
    
    # Initialize the MultiHeadAttention model
    model = Attention(emb_size)
    
    # Perform the forward pass and return the output
    output = model.forward((data,data,data))
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

