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
from typing import Optional

class MultiHeadAttention():
    """
    A Multi-head Attention module.

    This class can be used to perform multi-head attention operation on the input data.
    """

    def __init__(self, emb_size: int, num_heads: int):
        self.emb_size = emb_size
        self.num_heads = num_heads

        assert emb_size % num_heads == 0  # ensure divisibility of the dimensions

        self.Wq = np.random.randn(emb_size, emb_size)   # query weights
        self.Wk = np.random.randn(embize, emb_size)    # key weights
        self.Wv = np.random.randn(emb_size, emb_size)     # value weights

        self.d_k = int(self.emb_size / self.num_heads)  # dimension of each head

    def scaled_dot_product_attention(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        This function implements the scaled dot-product attention mechanism for multi-head self-attention

        Args:
            Q (np.ndarray): Query matrix of shape `(batch_size, seq_len, emb_size)`
            K (np.ndarray): Key matrix of shape `(batch_size, seq_len, emb_size)`
            V (np.ndarray): Value matrix of shape `(batch_size, seq_len, emb_size)`

        Returns:
            np.ndarray: Attention output of shape `(batch_size, num_heads, seq_len, d_k)`

        Note that the implementation here doesn't handle numeric overflow yet and applies softmax directly on raw scores.
        """

        # Scaled dot-product attention
        matmul_QK = np.matmul(Q, K.transpose(-2, -1))  # (batch_size, seq_len, seq_len)
        scaled_attention_logits = matmul_QK / np.sqrt(self.d_k)

        attention_weights = np.exp(scaled_attention_logits)/np.sum(np.exp(scaled_attention_logits), axis=-1, keepdims=True)  # apply softmax

        return np.matmul(attention_weights, V)

    def forward(self, Q: np.ndarray, K: Optional[np.ndarray] = None, V: Optional[np.ndarray] = None):
        """
         This function performs the multi-head attention mechanism and return output.

         Args:
            Q (np.ndarray): Query matrix of shape `(batch_size, seq_len, emb_size)`
            K (Optional[np.ndarray]): Key matrix of shape `(batch_size, seq_len, emb_size)`
            V (Optional[np.ndarray]): Value matrix of shape `(batch_size, seq_len, emb_size)`

        Returns:
           np.ndarray: Output tensor after applying multi-head attention mechanism. Shape is `(batch_size, num_heads, seq_len, d_k)`
        """

        if K is None and V is None:
            Q = Q @ self.Wq   # (batch_size, seq_len, emb_size)
            K = Q @ self.Wk    # (batch_size, seq_len, emb_size)
            V = Q @ self.Wv     # (batch_size, seq_len, emb_size)

        batch_size = Q.shape[0]  # get the first dimension size

        Q = Q.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(2, 0, 1, 3)
        K = K.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(2, 0, 1, 3)
        V = V.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(2, 0, 1, 3)

        output = self.scaled_dot_product_attention(Q, K, V)    # (num_heads, batch_size, seq_len, d_k)

        return output.transpose(1, 2, 0, 3).reshape(batch_size, -1, self.emb_size)   # reshape back to match original shape
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

