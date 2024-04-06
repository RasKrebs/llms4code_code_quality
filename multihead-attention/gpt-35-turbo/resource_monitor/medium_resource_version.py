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
    def __init__(self, emb_size: int, num_heads: int):
        assert emb_size % num_heads == 0, "emb_size must be divisible by num_heads"
        
        self.emb_size = emb_size
        self.num_heads = num_heads
        
        self.head_size = emb_size // num_heads
        
        # Initialize weight matrices for query, key, value, and output projections
        self.W_q = np.random.randn(emb_size, emb_size)
        self.W_k = np.random.randn(emb_size, emb_size)
        self.W_v = np.random.randn(emb_size, emb_size)
        self.W_o = np.random.randn(emb_size, emb_size)
        
    def scaled_dot_product_attention(self, Q, K, V):
        d_k = K.shape[-1]
        scaled_dot_product = np.matmul(Q, K.T) / np.sqrt(d_k)
        attention_weights = np.exp(scaled_dot_product - np.max(scaled_dot_product, axis=-1, keepdims=True))
        attention_weights /= np.sum(attention_weights, axis=-1, keepdims=True)
        output = np.matmul(attention_weights, V)
        return output
        
    def forward(self, X):
        batch_size, seq_length, emb_size = X.shape
        
        # Project input X into query, key, and value matrices
        Q = np.dot(X, self.W_q)
        K = np.dot(X, self.W_k)
        V = np.dot(X, self.W_v)
        
        # Split the matrices into multiple heads
        Q = np.reshape(Q, (batch_size, seq_length, self.num_heads, self.head_size))
        K = np.reshape(K, (batch_size, seq_length, self.num_heads, self.head_size))
        V = np.reshape(V, (batch_size, seq_length, self.num_heads, self.head_size))
        
        # Transpose the dimensions to perform attention over the sequence length
        Q = np.transpose(Q, (0, 2, 1, 3))
        K = np.transpose(K, (0, 2, 1, 3))
        V = np.transpose(V, (0, 2, 1, 3))
        
        # Apply scaled dot product attention for each head
        outputs = self.scaled_dot_product_attention(Q, K, V)
        
        # Reshape and concatenate the output heads
        outputs = np.transpose(outputs, (0, 2, 1, 3))
        outputs = np.reshape(outputs, (batch_size, seq_length, emb_size))
        
        # Project the concatenated output back to the original size
        outputs = np.dot(outputs, self.W_o)
        
        return outputs
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

