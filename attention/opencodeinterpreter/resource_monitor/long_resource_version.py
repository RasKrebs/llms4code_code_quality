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
        self.emb_size = emb_size
        # Initialize weights with random values
        self.weights = np.random.rand(emb_size*3)

    @staticmethod
    def scaled_dot_product_attention(query: np.ndarray, key: np.ndarray, value: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Performs Scaled Dot Product Attention between query and keys
        :param query: numpy array of shape (emb_size,)
        :param key: numpy array of shape (emb_size,)
        :param value: numpy array of shape (emb_size,)

        :return: Tuple containing the attention weighted value and the attention weights
        """
        # Compute dot product between query and keys
        dot_product = np.dot(query.T, key)

        # Scale by square root of emb_size (to avoid very small numbers when doing softmax)
        scaled_dot_product = dot_product / np.sqrt(len(key))

        # Apply softmax to get attention weights
        attention_weights = np.exp(scaled_dot_product - np.max(scaled_dot_product)) / \
                              np.sum(np.exp(scaled_dot_product - np.max(scaled_dot_product)))

        # Compute weighted sum of values using attention weights
        weighted_value = np.dot(attention_weights, value.T)

        return weighted_value, attention_weights

    def forward(self, input_tensor: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Tuple[np.ndarray, float]:
        """
        Forward pass of Attention mechanism

        :param input_tensor: A tuple containing the query, key and value tensors
        Each tensor is a numpy array of shape (emb_size,)

        :return: Tuple containing the attention weighted value and the attention weights
        """
        # Separate the query, key and value from input_tensor
        query, key, value = input_tensor

        # Perform Scaled Dot Product Attention
        weighted_value, attention_weights = self.scaled_dot_product_attention(query, key, value)

        return weighted_value, attention_weights

def execute(emb_size, num_heads):
    # Load the data
    data = np.random.rand(emb_size, seq_length)
    
    # Initialize the MultiHeadAttention model
    model = Attention(emb_size)
    
    # Perform the forward pass and return the output
    output = model.forward((data, data, data))
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

