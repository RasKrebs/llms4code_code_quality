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
from scipy.special import softmax

class MultiHeadAttention:
    def __init__(self, embed_size=512, num_heads=8):
        self.embed_size = embed_size
        self.num_heads = num_heads

        assert (self.embed_size % self.num_heads) == 0

        # Note that we don't end up backpropagating over any of the individual components, because they are constant for all inputs in one pass
        self.d_k = embed_size // num_heads  # Embedding dimension per head
        self.query = np.random.randn(embed_size, self.d_k)
        self.key = np.random.randn(embed_size, self.d_k)
        self.value = np.random.randn(embed_size, self.d_k)

    def forward(self, x):
        # Reshape the input tensor to [batch size, sequence length, embed size]
        batch_size, sequence_length, _ = x.shape
        reshaped_x = np.reshape(x, (batch_size, sequence_length, self.num_heads, self.d_k))

        # Compute Q, K, V for each head
        Q = np.dot(reshaped_x, self.query)  # [batch size, sequence length, num heads, d_k]
        K = np.dot(reshaped_x, self.key)  # [batch size, sequence length, num heads, d_k]
        V = np.dot(reshaped_x, self.value)  # [batch size, sequence length, num heads, d_k]

        # Compute scaled dot product attention for each head
        scores = np.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)  # [batch size, num heads, sequence length, sequence length]
        weights = softmax(scores, axis=-1)  # [batch size, num heads, sequence length, sequence length]

        attention = np.matmul(weights, V)  # [batch size, num heads, sequence length, d_k]
        result = np.concatenate(np.split(attention, self.num_heads, axis=2), axis=-1)  # [batch size, sequence length, embed size]

        return result
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

