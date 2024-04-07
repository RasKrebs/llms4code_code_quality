import psutil
import os
import threading
import multiprocessing

# Define a global variable to store the maximum resources usage
max_resources_usage = {"cpu": 0, "memory": 0}



import numpy as np

np.random.seed(42)
seq_length = 100 
emb_size = 512


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

class Attention:
    """
    Attention class for transformer models.
    """
    def __init__(self, emb_size: int):
        """
        Initialize with emb_size (input embedding size).
        Initialize weights with random values.
        """
        self.emb_size = emb_size
        self.weights = [random.random() for _ in range(emb_size)]

    def scaled_dot_product_attention(self, query, key, value):
        """
        Perform the attention operation, performing scaled_dot_product,
        which also applied softmax for stability.
        """
        query = np.array(query)
        key = np.array(key)
        value = np.array(value)

        matmul_qk = np.dot(query, key.T)

        # scale matmul_qk
        dk = math.sqrt(len(query))
        scaled_attention_logits = matmul_qk / dk
        # softmax is applied for stability
        attention_weights = np.exp(scaled_attention_logits.T) / np.sum(np.exp(scaled_attention_logits).T, axis=1, keepdims=True)

        output = np.dot(value.T, attention_weights)

        return output, attention_weights

    def forward(self, input_tensor):
        """
        The forward method projects input tensors into query, key, and value tensors
        using the initialized weights and performs attention.
        """
        query = [i * w for i, w in zip(input_tensor, self.weights)]
        key = [i * w for i, w in zip(input_tensor, self.weights)]
        value = [i * w for i, w in zip(input_tensor, self.weights)]

        return self.scaled_dot_product_attention(query, key, value)

def execute(emb_size, num_heads):
    # Load the data
    data = np.random.rand(seq_length, emb_size)
    
    # Initialize the MultiHeadAttention model
    model = Attention(emb_size)
    
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
    output = execute(emb_size,seq_length)


    # Stop the monitoring
    monitoring = False
    monitor_thread.join()

    print(max_resources_usage['cpu']), print(max_resources_usage['memory'] / (1024**2))

