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
from typing import List

class MultiHeadAttention:
    """Implementation of a Multi-head Attention Mechanism."""

    def __init__(self, emb_size: int, num_heads: int):
        self.emb_size = emb_size
        self.num_heads = num_heads

        # Calculate the dimension of each head
        self.dim_per_head = emb_size // num_heads
        assert self.dim_per_head * num_heads == emb_size, "emb_size must be divisible by num_heads"

        # Initialize random weight matrices for query, key, value and output projections
        self.query = np.random.randn(self.num_heads, emb_size)
        self.key = np.random.randn(self.num_heads, emb_size)
        self.value = np.random.randn(self.num_heads, emb_size)
        self.output = np.random.randn(emb_size, num_heads)

    def scaled_dot_product_attention(self, query: List[float], key: List[float], value: List[float]) -> float:
        """Perform the Scaled Dot-Product Attention operation."""
        scores = np.dot(query, key) / np.sqrt(len(key))  # Scale dot product by square root of dimension

        # Prevent numerical overflow during softmax computation
        max_value = np.max(scores)
        exp_scores = np.exp(scores - max_value)
        softmaxed_scores = exp_scores / np.sum(exp_scores)  # Apply Softmax

        weighted_values = np.dot(softmaxed_scores, value)

        return weighted_values

    def forward(self, inputs: List[float]) -> float:
        """Perform the Multi-head Attention operation."""
        # Linear transformations for queries, keys and values
        Q = np.dot(inputs, self.query)
        K = np.dot(inputs, self.key)
        V = np.dot(inputs, self.value)

        attn_outputs = []

        # Perform Scaled Dot-Product Attention for each head and concatenate results
        for q, k, v in zip(Q, K, V):
            attn_output = self.scaled_dot_product_attention(q, k, v)
            attn_outputs.append(attn_output)

        # Concatenate the attention outputs from all heads and apply output projection
        concat_output = np.concatenate(attn_outputs)

        return np.dot(concat_output, self.output)
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

