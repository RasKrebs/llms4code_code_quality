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
    def __init__(self, embedding_size, num_heads):
        assert embedding_size % num_heads == 0, "Embedding size must be divisible by number of heads."

        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.depth = embedding_size // num_heads
        
        # Initialize weights for Q, K, V, and the final linear layer
        self.W_Q = np.random.randn(embedding_size, embedding_size)
        self.W_K = np.random.randn(embedding_size, embedding_size)
        self.W_V = np.random.randn(embedding_size, embedding_size)
        self.W_O = np.random.randn(embedding_size, embedding_size)
    
    def split_heads(self, x):
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1, self.num_heads, self.depth).transpose(0, 2, 1, 3)
    
    def scaled_dot_product_attention(self, Q, K, V):
        matmul_qk = np.matmul(Q, K.transpose(0,1,3,2))
        
        # Scale the product
        dk = np.sqrt(K.shape[-1])
        scaled_attention_logits = matmul_qk / dk
        
        # Apply softmax
        attention_weights = np.exp(scaled_attention_logits) / np.sum(np.exp(scaled_attention_logits), axis=-1, keepdims=True)
        
        # Weighted sum to get output
        output = np.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        Q = np.dot(x, self.W_Q)
        K = np.dot(x, self.W_K)
        V = np.dot(x, self.W_V)
        
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        attention_output, _ = self.scaled_dot_product_attention(Q, K, V)
        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.embedding_size)
        
        output = np.dot(attention_output, self.W_O)
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

