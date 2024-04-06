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
from typing import Tuple

class MultiHeadAttention:
    def __init__(self, emb_size: int, num_heads: int):
        """
        Initialize a new Multi-Head Attention layer.

        Args:
            emb_size (int): The input embedding size which should be divisible by the number of heads.
            num_heads (int): The number of attention heads.

        Raises:
            ValueError if emb_size is not divisible by num_heads.
        """

        if emb_size % num_heads != 0:
            raise ValueError(f"Embedding size ({emb_size}) must be divisible by the number of heads ({num_heads}).")

        self.emb_size = emb_size
        self.num_heads = num_heads

        # Weights for query, key and value projections
        self.q_weights = np.random.randn(emb_size, emb_size)
        self.k_weights = np.random.randn(emb_size, emb_size)
        self.v_weights = np.random.randn(emb_size, emb_size)

        # Weights for the output projection
        self.o_weights = np.random.randn(num_heads * (emb_size // num_heads), emb_size)

    def scaled_dot_product_attention(self, q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the scaled dot product attention for a single head.

        Args:
            q (numpy array): The query tensor of shape (seq_len, emb_size).
            k (numpy array): The key tensor of shape (seq_len, emb_size).
            v (numpy array): The value tensor of shape (seq_len, emb_size).
            mask (numpy array): A binary mask for padding elements. Defaults to None.

        Returns:
            output (numpy array): The result tensor of shape (seq_len, emb_size // num_heads).
            attention_weights (numpy array): The calculated attention weights of shape (seq_len, seq_len).
        """

        # Calculate the dot product between query and key, scaled by sqrt(emb_size) for stability
        matmul_qk = np.matmul(q, k.transpose())
        d_k = q.shape[-1] ** 0.5

        # Applying the scaling factor
        scaled_attention_logits = matmul_qk / d_k

        if mask is not None:
            # Add a very small number to the logits to avoid division by zero in softmax
            scaled_attention_logits += (mask * -1e9)

        # Calculate attention weights
        attention_weights = np.exp(scaled_attention_logits - np.max(scaled_attention_logits, axis=-1, keepdims=True))
        attention_weights /= np.sum(attention_weights, axis=-1, keepdims=True)

        # Apply the attention weights to the value tensor
        output = np.matmul(attention_weights, v)

        return output, attention_weights

    def forward(self, x: np.ndarray, mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
       ^[ Perform the forward pass through Multi-Head Attention.

        Args:
            x (numpy array): The input tensor of shape (seq_len, emb_size).
            mask (numpy array): A binary mask for padding elements. Defaults to None.

        Returns:
            output (numpy array): The result tensor after concatenating all heads and applying the output projection, of shape (seq_len, emb_size).
            attention_weights (numpy array): The calculated attention weights for each head, stacked along dimension 0. Shape is (num_heads, seq_len, seq_len)
        """

        # Calculate the sizes of the dimensions
        seq_len = x.shape[1]
        emb_size_per_head = self.emb_size // self.num_heads

        # Split embedding into multiple heads
        q, k, v = [np.reshape(np.matmul(x, weights), (seq_len, -1, emb_size_per_head)) for weights in [self.q_weights, self.k_weights, self.v_weights]]

        # Compute attention output and weights for each head
        outputs = []
        attention_weights = []
        for i in range(self.num_heads):
            o, aw = self.scaled_dot_product_attention(q[i], k[i], v[i], mask)
            outputs.append(o)
            attention_weights.append(aw)

        # Concatenate the results from all heads and apply output projection
        concat = np.concatenate(outputs, axis=-1)
        output = np.matmul(concat, self.o_weights)

        return output, np.stack(attention_weights)
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

