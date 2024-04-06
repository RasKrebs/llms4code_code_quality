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
from typing import Tuple

class ScaledDotProductAttention:
    def forward(self, query, key, value):
        dk = query.shape[-1]

        # Perform scaled dot product attention
        scores = np.matmul(query, key.transpose()) / np.sqrt(dk)

        # Handle numeric overflow by subtracting the maximum value for each row from all elements in that row
        max_values = np.amax(scores, axis=-1, keepdims=True)
        exp_input = scores - max_values

        # Apply softmax and dropout
        attention_weights = np.exp(exp_input) / np.sum(np.exp(exp_input), axis=-1, keepdims=True)

        output = np.matmul(attention_weights, value)

        return output

class MultiHeadAttention:
    def __init__(self, emb_size: int, num_heads: int):
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads  # Calculate the dimension of each head ensuring divisibility

        assert (self.head_dim * num_heads == emb_size), "emb_size needs to be divisible by num_heads"

        self.qkv_projection = np.random.rand(3*emb_size, emb_size)  # Randomly initialize weight matrices for query, key, value projections
        self.out_projection = np.random.rand(emb_size, emb_size)  # Randomly initialize output projection matrix

        self.attention = ScaledDotProductAttention()

    def split_heads(self, x):
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose((0, 2, 1, 3))

    def forward(self, inputs):
        batch_size = inputs.shape[0]

        # Apply linear transformations to get queries, keys and values
        qkv = np.dot(inputs, self.qkv_projection)
        query, key, value = np.split(qkv, 3, axis=-1)

        # Split the heads for multi-head attention
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        # Perform scaled dot product attention and concatenate the heads
        output = np.concatenate([self.attention.forward(q, k, v) for q, k, v in zip(query, key, value)], axis=-1)

        # Apply final linear transformation
        output = np.dot(output, self.out_projection)

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

