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



import math
from typing import Tuple

class MultiHeadAttention:
    def __init__(self, emb_size: int, num_heads: int):
        assert emb_size % num_heads == 0, "emb_size should be divisible by num_heads"

        self.emb_size = emb_size
        self.num_heads = num_heads
        self.depth = emb_size // num_heads

        # Initialize weights for query, key, value, and output projections
        self.query = [[math.random()] * self.emb_size for _ in range(self.num_heads)]
        self.key = [[math.random()] * self.emb_size for _ in range(self.num_heads)]
        self.value = [[math.random()] * self.emb_size for _ in range(self.num_heads)]
        self.output = [[math.random()] * emb_size for _ in range(self.num_heads)]

    def scaled_dot_product_attention(self, query: Tuple[int], key: Tuple[int], value: Tuple[int]) -> Tuple[float]:
        dot_product = sum(q * k for q, k in zip(query, key)) / math.sqrt(len(query))
        softmax = lambda x : math.exp(x) / sum(math.exp(i) for i in x)

        attention_weights = [softmax(dot_product)] * len(value)

        weighted_sum = [w * v for w, v in zip(attention_weights, value)]

        return tuple(weighted_sum)

    def forward(self, input: Tuple[Tuple[int]]) -> Tuple[float]:
        # Project input into query, key, and value tensors using the initialized weights
        queries = [tuple(sum(q * i for q, i in zip(query_weight, input_)) for input_ in input) for query_weight in self.query]
        keys = [tuple(sum(k * i for k, i in zip(key_weight, input_)) for input_ in input) for key_weight in self.key]
        values = [tuple(sum(v * i for v, i in zip(value_weight, input_)) for input_ in input) for value_weight in self.value]

        # Apply attention operation to each head
        attention_outputs = []
        for query_, key_, value_ in zip(queries, keys, values):
            attention_result = self.scaled_dot_product_attention(query_, key_, value_)
            attention_outputs.append(attention_result)

        # Concatenate the results from each head and apply an output projection
        concat_attention = [sum(i[j] for i in attention_outputs) for j in range(len(input[0]))]
        output = tuple(sum(o * i for o, i in zip(output_weight, concat_attention)) for output_weight in self.output)

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

