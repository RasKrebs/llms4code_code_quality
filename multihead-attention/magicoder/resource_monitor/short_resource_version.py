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

class AttentionHead:
    def __init__(self, embedding_size):
        self.Wq = np.random.randn(embedding_size, embedding_size)  # query weight
        self.Wk = np.random.randn(embedding_size, embedding_size)  # key weight
        self.Wv = np.random.randn(embedding_size, embedding_size)  # value weight

    def __call__(self, query, key):
        Q = np.matmul(query, self.Wq)
        K = np.matmul(key, self.Wk)
        V = np.matmul(key, self.Wv)  # value is also the output of attention mechanism

        scores = np.matmul(Q, K.T) / np.sqrt(embedding_size)  # scaled dot-product attention
        weights = np.exp(scores - np.max(scores)) / np.sum(np.exp(scores - np.max(scores)))  # softmax operation

        return np.matmul(weights, V)

class MultiHeadAttention:
    def __init__(self, embedding_size, num_heads):
        self.embedding_size = embedding_size
        self.num_heads = num_heads

        self.attention_heads = [AttentionHead(embedding_size) for _ in range(num_heads)]

    def __call__(self, query, key):
        return np.concatenate([ah(query, key) for ah in self.attention_heads])  # concatenating the results of each attention head
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

