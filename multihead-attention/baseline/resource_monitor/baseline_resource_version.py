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



"""
This module contains the implementation of the MultiHeadAttention layer. The code draws
inspiration from a mixture of sources. 
"""

import numpy as np


class MultiHeadAttention:
    """Multi-head attention layer.

    Attributes:
        emb_size (int): The size of the input embedding.
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        weights_q (ndarray): The weights for the query projection.
        weights_k (ndarray): The weights for the key projection.
        weights_v (ndarray): The weights for the value projection.
        weights_o (ndarray): The weights for the output projection.
    """

    def __init__(self, emb_size:int, num_heads:int):
        """Initializes the MultiHeadAttention layer.

        Args:
            emb_size (int): The size of the input embedding.
            num_heads (int): The number of attention heads.
        """
        # Save attributes
        self.emb_size = emb_size
        self.num_heads = num_heads
        
        # Check if embedding size is divisible by number of heads
        self.head_dim = emb_size // num_heads
        if not self.head_dim * num_heads == emb_size:
            raise ValueError("Embedding size must be divisible by number of heads")

        # Initialize weights for query, key, value for each head
        self.weights_q = np.random.rand(num_heads, emb_size, self.head_dim)
        self.weights_k = np.random.rand(num_heads, emb_size, self.head_dim)
        self.weights_v = np.random.rand(num_heads, emb_size, self.head_dim)

        # Final linear projection weights
        self.weights_o = np.random.rand(emb_size, emb_size)

    def scaled_dot_product_attention(self, query, key, value):
        """Performs scaled dot-product attention.

        Args:
            query (ndarray): The query tensor.
            key (ndarray): The key tensor.
            value (ndarray): The value tensor.

        Returns:
            ndarray: The output tensor after attention.
        """
        matmul_qk = np.matmul(query, key.transpose((0, 2, 1)))
        depth = query.shape[-1]
        logits = matmul_qk / np.sqrt(depth)

        # Normalize logits to prevent overflow
        logits -= np.max(logits, axis=-1, keepdims=True)

        weights = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
        output = np.matmul(weights, value)
        return output

    def forward(self, data):
        """Performs forward pass of the MultiHeadAttention layer.

        Args:
            x (ndarray): The input tensor.

        Returns:
            ndarray: The output tensor after attention.
        """
        batch_size = data.shape[0]

        # Linear projection split into heads
        query = np.concatenate([np.dot(data, self.weights_q[i]) for i in range(self.num_heads)],
                               axis=0)
        key = np.concatenate([np.dot(data, self.weights_k[i]) for i in range(self.num_heads)],
                             axis=0)
        value = np.concatenate([np.dot(data, self.weights_v[i]) for i in range(self.num_heads)],
                               axis=0)

        # Reshape for heads
        query = query.reshape(batch_size, self.num_heads, -1, self.head_dim)
        key = key.reshape(batch_size, self.num_heads, -1, self.head_dim)
        value = value.reshape(batch_size, self.num_heads, -1, self.head_dim)

        # Scaled dot-product attention for each head
        heads = []
        for i in range(self.num_heads):
            heads.append(self.scaled_dot_product_attention(query[:, i], key[:, i], value[:, i]))
        head = np.concatenate(heads, axis=-1)

        # Final linear projection
        output = np.dot(head.reshape(batch_size, -1), self.weights_o)
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

