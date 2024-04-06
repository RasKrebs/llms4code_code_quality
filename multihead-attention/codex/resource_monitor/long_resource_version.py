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
import random

class MultiHeadAttention:
    def __init__(self, emb_size: int, num_heads: int):
        """
        Initialize the MultiHeadAttention class.
        """
        if emb_size % num_heads != 0:
            raise ValueError("emb_size must be divisible by num_heads")

        self.emb_size = emb_size
        self.num_heads = num_heads

        # Initialize weights with random values
        self.query_weights = [[random.random() for _ in range(emb_size)] for _ in range(emb_size)]
        self.key_weights = [[random.random() for _ in range(emb_size)] for _ in range(emb_size)]
        self.value_weights = [[random.random() for _ in range(emb_size)] for _ in range(emb_size)]
        self.output_weights = [[random.random() for _ in range(emb_size)] for _ in range(emb_size)]

    # Changed a bit
    def scaled_dot_product_attention(self, query: list, key: list, value: list):
        # Compute the dot products between all queries and keys
        dot_products = [[sum(q * k for q, k in zip(q_row, k_col)) for k_col in zip(*key)] for q_row in query]

        # Scale dot products by sqrt of the dimension of key vectors
        depth = len(key[0])
        scaled_dot_products = [[dp / math.sqrt(depth) for dp in row] for row in dot_products]

        # Apply softmax to each row of the scaled dot products
        softmax_outputs = [self._softmax(row) for row in scaled_dot_products]

        # Multiply softmax outputs by the value matrix and sum up
        output = []
        for softmax_row in softmax_outputs:
            weighted_sum = [0] * len(value[0])
            for i, softmax_val in enumerate(softmax_row):
                weighted_sum = [sum_val + softmax_val * val[i] for sum_val, val in zip(weighted_sum, value)]
            output.append(weighted_sum)

        return output

    def _softmax(self, x):
        """Compute softmax values for each set of scores in x."""
        e_x = np.exp(x - np.max(x))  # Stability improvement by subtracting max from x
        sum_e_x = np.sum(e_x)
        return e_x / sum_e_x


    def forward(self, input_tensor: list):
        """
        Forward pass of the MultiHeadAttention.
        """
        # Project input tensor into query, key, and value tensors
        query = [sum(i * w for i, w in zip(input_tensor, row)) for row in self.query_weights]
        key = [sum(i * w for i, w in zip(input_tensor, row)) for row in self.key_weights]
        value = [sum(i * w for i, w in zip(input_tensor, row)) for row in self.value_weights]

        # Apply attention
        attention = self.scaled_dot_product_attention(query, key, value)

        # Concatenate results
        concatenated = [a + i for a, i in zip(attention, input_tensor)]

        # Apply output projection
        output = [sum(i * w for i, w in zip(concatenated, row)) for row in self.output_weights]

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

