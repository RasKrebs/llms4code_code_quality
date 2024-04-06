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

class MultiHeadAttention:
    def __init__(self, embedding_dim, num_heads):
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.depth = embedding_dim // num_heads

        assert self.depth * num_heads == self.embedding_dim, "Embedding dimension must be divisible by number of heads"

        self.wq = np.random.rand(embedding_dim, embedding_dim)
        self.wk = np.random.rand(embedding_dim, embedding_dim)
        self.wv = np.random.rand(embedding_dim, embedding_dim)

    def split_heads(self, x):
        batch_size = x.shape[0]
        x = np.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return np.transpose(x, (0, 2, 1, 3))

    def scaled_dot_product_attention(self, q, k, v):
        matmul_qk = np.matmul(q, np.transpose(k, (0, 1, 3, 2)))
        dk = np.float32(k.shape[-1])
        scaled_attention_logits = matmul_qk / np.sqrt(dk)
        attention_weights = np.exp(scaled_attention_logits) / np.sum(np.exp(scaled_attention_logits), axis=-1, keepdims=True)
        output = np.matmul(attention_weights, v)
        return output, attention_weights

    def forward(self, v, k, q):
        batch_size = q.shape[0]

        q = np.matmul(q, self.wq)
        k = np.matmul(k, self.wk)
        v = np.matmul(v, self.wv)

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v)

        scaled_attention = np.transpose(scaled_attention, (0, 2, 1, 3))
        concat_attention = np.reshape(scaled_attention, (batch_size, -1, self.embedding_dim))

        return concat_attention

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
    model = MultiHeadAttention(emb_size, num_heads)
    
    # Generate random query, key, and value tensors
    query = np.random.rand(batch_size, seq_length, emb_size)
    key = np.random.rand(batch_size, seq_length, emb_size)
    value = np.random.rand(batch_size, seq_length, emb_size)
    
    # Perform the forward pass and return the output
    output = model.forward(value, key, query)
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

