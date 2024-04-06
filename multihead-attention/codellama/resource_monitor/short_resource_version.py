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

class MultiHeadAttention(object):
    def __init__(self, embedding_size, num_heads):
        self.embedding_size = embedding_size
        self.num_heads = num_heads

        # Generate random weights for the query, key, and value matrices
        self.Wq = [[random.random() for _ in range(self.embedding_size)] for _ in range(self.embedding_size)]
        self.Wk = [[random.random() for _ in range(self.embedding_size)] for _ in range(self.embedding_size)]
        self.Wv = [[random.random() for _ in range(self.embedding_size)] for _ in range(self.embedding_size)]

    def forward(self, input):
        # Check if the number of heads divides the embedding size evenly
        assert self.num_heads % self.embedding_size == 0, "Number of heads must be divisible by the embedding size"

        # Calculate the depth of each head (embedding_size / num_heads)
        dk = self.embedding_size // self.num_heads

        # Split the input tensor into multiple heads and apply attention
        output = [self.attention(input, input, input) for _ in range(self.num_heads)]

        # Concatenate the outputs of each head
        concat_output = []
        for row in output:
            concat_row = []
            for element in row:
                concat_row += element
            concat_output.append(concat_row)

        return concat_output

    def attention(self, query, key, value):
        # Calculate the query-key product and normalize it with a softmax function
        qk = [[sum([query[i][j] * key[i][j] for j in range(len(key[i]))]) for i in range(len(key))] for i in range(len(value))]
        attn_weights = self.softmax(qk)

        # Apply the attention weights to the value vector and calculate the output
        output = [[sum([attn_weights[i][j] * value[j][k] for j in range(len(value))]) for k in range(len(key))] for i in range(len(query))]

        return output

    def softmax(self, x):
        # Calculate the softmax of a vector
        e_x = [math.exp(i) for i in x]
        sum_e_x = sum(e_x)
        return [i / sum_e_x for i in e_x]
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

