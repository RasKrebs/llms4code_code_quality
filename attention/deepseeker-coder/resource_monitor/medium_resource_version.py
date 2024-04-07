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



import numpy as np
from typing import Tuple

class Attention:
    def __init__(self, emb_size: int):
        """
        Initialize the Attention mechanism.

        Parameters:
        emb_size (int): The size of the embedding vectors.
        """
        self.emb_size = emb_size
        # Initialize weight matrices for query, key, value, and output projections
        self.W_q = np.random.randn(emb_size, emb_size)
        self.W_k = np.random.randn(emb_size, emb_size)
        self.W_v = np.random.randn(emb_size, emb_size)
        self.W_o = np.random.randn(emb_size, emb_size)

    def scaled_dot_product_attention(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the scaled dot-product attention.

        Parameters:
        Q (np.ndarray): Query matrix of shape (emb_size, seq_length)
        K (np.ndarray): Key matrix of shape (emb_size, seq_length)
        V (np.ndarray): Value matrix of shape (emb_size, seq_length)

        Returns:
        output (np.ndarray): Output matrix of shape (emb_size, seq_length)
        attn_weights (np.ndarray): Attention weights matrix of shape (seq_length, seq_length)
        """
        # Compute the dot product of Q and K^T
        dot_product = np.dot(Q, K.T)

        # Scale the dot product by the square root of the embedding size
        scale = np.sqrt(self.emb_size)
        scaled_dot_product = dot_product / scale

        # Apply softmax to get the attention weights
        attn_weights = self.softmax(scaled_dot_product)

        # Compute the output as the weighted sum of the value vectors
        output = np.dot(attn_weights, V)

        return output, attn_weights

    def softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the softmax of a matrix.

        Parameters:
        x (np.ndarray): Input matrix

        Returns:
        softmax_x (np.ndarray): Softmax matrix
        """
        # Subtract the max of x for numerical stability
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform the forward pass of the attention mechanism.

        Parameters:
        x (np.ndarray): Input matrix of shape (emb_size, seq_length)

        Returns:
        output (np.ndarray): Output matrix of shape (emb_size, seq_length)
        attn_weights (np.ndarray): Attention weights matrix of shape (seq_length, seq_length)
        """
        # Compute the query, key, and value matrices
        Q = np.dot(self.W_q, x)
        K = np.dot(self.W_k, x)
        V = np.dot(self.W_v, x)

        # Compute the output and attention weights
        output, attn_weights = self.scaled_dot_product_attention(Q, K, V)

        # Apply the output projection
        output = np.dot(self.W_o, output)

        return output, attn_weights

def execute(emb_size, num_heads):
    # Load the data
    data = np.random.rand(emb_size, seq_length)
    
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

