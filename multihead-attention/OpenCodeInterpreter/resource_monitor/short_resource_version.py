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
    def __init__(self, embedding_size: int, n_heads: int):
        self.embedding_size = embedding_size
        self.n_heads = n_heads

        # Create weight matrices for each head (query, key and value)
        self.weights_q = [np.random.rand(embedding_size, embedding_size//n_heads) for _ in range(n_heads)]
        self.weights_k = [np.random.rand(embedding_size, embedding_size//n_heads) for _ in range(n_heads)]
        self.weights_v = [np.random.rand(embedding_size, embedding_size//n_heads) for _ in range(n_heads)]

    def forward(self, input_tensor: List[List[int]]) -> List[List[float]]:
        output_tensors = []

        # Perform attention for each head
        for i in range(self.n_heads):
            # Apply linear transformations to the input tensor
            q = np.matmul(input_tensor, self.weights_q[i])  # Query
            k = np.matmul(input_tensor, self.weights_k[i])  # Key
            v = np.matmul(input_tensor, self.weights_v[i])  # Value

            # Compute attention scores
            attn_scores = np.matmul(q, k.T) / np.sqrt(self.embedding_size//n_heads)

            # Apply softmax to attention scores
            attn_probs = np.exp(attn_scores - np.max(attn_scores))
            attn_probs /= np.sum(attn_probs, axis=1, keepdims=True)

            # Apply attention weights to values
            weighted_values = np.matmul(attn_probs, v)

            output_tensors.append(weighted_values)

        return output_tensors
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

