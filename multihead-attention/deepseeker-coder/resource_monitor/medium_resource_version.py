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
import math
from typing import Tuple

class MultiHeadAttention:
    def __init__(self, emb_size: int, num_heads: int):
        """
        Initialize the MultiHeadAttention module.

        Parameters:
        emb_size (int): The size of the embedding.
        num_heads (int): The number of attention heads.
        """
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_size = emb_size // num_heads

        # Randomly initialize weight matrices for query, key, value, and output projections
        self.query_weights = np.random.randn(emb_size, emb_size)
        self.key_weights = np.random.randn(emb_size, emb_size)
        self.value_weights = np.random.randn(emb_size, emb_size)
        self.output_weights = np.random.randn(emb_size, emb_size)

    def scaled_dot_product_attention(self, query: np.ndarray, key: np.ndarray, value: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the attention weights and the output of the scaled dot-product attention.

        Parameters:
        query (np.ndarray): The query matrix.
        key (np.ndarray): The key matrix.
        value (np.ndarray): The value matrix.

        Returns:
        Tuple[np.ndarray, np.ndarray]: The attention weights and the output of the scaled dot-product attention.
        """
        # Calculate the dot product of the query and key matrices
        dot_product = np.dot(query, key.T)

        # Scale the dot product by the square root of the embedding size
        scale = math.sqrt(self.emb_size)
        dot_product /= scale

        # Apply softmax to get the attention weights
        attention_weights = np.exp(dot_product) / np.sum(np.exp(dot_product), axis=-1, keepdims=True)

        # Calculate the output of the scaled dot-product attention
        output = np.dot(attention_weights, value)

        return attention_weights, output

    def forward(self, query: np.ndarray, key: np.ndarray, value: np.ndarray) -> np.ndarray:
        """
        Perform the multi-head attention and return the output.

        Parameters:
        query (np.ndarray): The query matrix.
        key (np.ndarray): The key matrix.
        value (np.ndarray): The value matrix.

        Returns:
        np.ndarray: The output of the multi-head attention.
        """
        # Apply the query, key, and value projections
        query = np.dot(query, self.query_weights)
        key = np.dot(key, self.key_weights)
        value = np.dot(value, self.value_weights)

        # Split the query, key, and value matrices into multiple heads
        query = query.reshape(-1, self.num_heads, self.head_size)
        key = key.reshape(-1, self.num_heads, self.head_size)
        value = value.reshape(-1, self.num_heads, self.head_size)

        # Perform the scaled dot-product attention for each head
        attention_weights, output = self.scaled_dot_product_attention(query, key, value)

        # Concatenate the output of each head
        output = output.reshape(-1, self.emb_size)

        # Apply the output projection
        output = np.dot(output, self.output_weights)

        return output
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

