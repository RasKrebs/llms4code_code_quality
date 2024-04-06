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
    """
    A class to perform multi-head attention mechanism.
    """
    def __init__(self, emb_size: int, num_heads: int):
        """
        Initialize the class with the given embedding size and number of heads.

        Args:
        emb_size (int): The size of the embedding.
        num_heads (int): The number of attention heads.
        """
        assert emb_size % num_heads == 0, "Embedding size must be divisible by number of heads"

        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads

        # Initialize weight matrices for query, key, value, and output projections
        self.W_q = np.random.rand(emb_size, emb_size)
        self.W_k = np.random.rand(emb_size, emb_size)
        self.W_v = np.random.rand(emb_size, emb_size)
        self.W_o = np.random.rand(emb_size, emb_size)

    def scaled_dot_product_attention(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        Perform the scaled dot-product attention mechanism.

        Args:
        Q (np.ndarray): The query matrix.
        K (np.ndarray): The key matrix.
        V (np.ndarray): The value matrix.

        Returns:
        np.ndarray: The result of the attention mechanism.
        """
        # Calculate the dot product of Q and K, and scale by the square root of the head dimension
        scaled_dot_product = np.matmul(Q, np.transpose(K, (0, 2, 1))) / np.sqrt(self.head_dim)

        # Apply softmax to the scaled dot product
        attention_weights = softmax(scaled_dot_product, axis=-1)

        # Multiply the attention weights by V to get the output
        output = np.dot(attention_weights, V)

        return output

    def forward(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        Perform the multi-head attention mechanism and return the output.

        Args:
        Q (np.ndarray): The query matrix.
        K (np.ndarray): The key matrix.
        V (np.ndarray): The value matrix.

        Returns:
        np.ndarray: The result of the multi-head attention mechanism.
        """
        # Project the input matrices using the weight matrices
        Q_proj = np.dot(Q, self.W_q)
        K_proj = np.dot(K, self.W_k)
        V_proj = np.dot(V, self.W_v)

        # Split the projected matrices into multiple heads
        Q_split = np.split(Q_proj, self.num_heads, axis=0)
        K_split = np.split(K_proj, self.num_heads, axis=0)
        V_split = np.split(V_proj, self.num_heads, axis=0)

        # Perform the scaled dot-product attention for each head
        outputs = [self.scaled_dot_product_attention(Q_head, K_head, V_head) for Q_head, K_head, V_head in zip(Q_split, K_split, V_split)]

        # Concatenate the outputs and project using the output weight matrix
        output_concat = np.concatenate(outputs, axis=0)
        output = np.dot(output_concat, self.W_o)

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
    

    # Generate random query, key, and value tensors
    query = np.random.rand(batch_size, seq_length, emb_size)
    key = np.random.rand(batch_size, seq_length, emb_size)
    value = np.random.rand(batch_size, seq_length, emb_size)
    
    # Perform the forward pass and return the output
    output = model.forward(query, key, value)
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

