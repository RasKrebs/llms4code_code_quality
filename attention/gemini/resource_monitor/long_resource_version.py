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

class Attention:
    """
    Class for implementing scaled dot-product attention from scratch for transformer models.
    """

    def __init__(self, emb_size: int):

        """
        Initializes the Attention layer.

        Args:
            emb_size: Embedding size of the input tensors.
        """
        super().__init__()
        self.emb_size = emb_size

        # Initialize weights with random values using a Gaussian distribution
        self.query_weights = np.random.randn(emb_size, emb_size)
        self.key_weights = np.random.randn(emb_size, emb_size)
        self.value_weights = np.random.randn(emb_size, emb_size)

        # Handle numerical instability for softmax
        self.eps = 1e-8

    def scaled_dot_product_attention(self, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Performs scaled dot-product attention.

        Args:
            q: Query tensor of shape (sequence_length, emb_size).
            k: Key tensor of shape (sequence_length, emb_size).
            v: Value tensor of shape (sequence_length, emb_size).

        Returns:
            The computed attention output tensor of shape (sequence_length, emb_size).
        """

        # Calculate attention scores (dot product of queries and keys scaled by square root of embedding size)
        scores = np.matmul(q, k.T) / np.sqrt(self.emb_size)  # (sequence_length, sequence_length)

        # Apply softmax to attention scores for normalization
        attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True)) + self.eps  # Numerical stability
        attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)

        # Calculate attention output by multiplying attention weights with values
        output = np.matmul(attn_weights, v)  # (sequence_length, emb_size)

        return output

    def forward(self, x: np.ndarray, sequence_length: int) -> np.ndarray:
        """
        Projects the input tensor into query, key, and value tensors using learned weights,
        and performs the scaled dot-product attention.

        Args:
            x: Input tensor of shape (sequence_length, emb_size).
            sequence_length: Length of the input sequence.

        Returns:
            The output tensor after attention of shape (sequence_length, emb_size).
        """

        # Project input into query, key, and value tensors
        q = np.matmul(x.T, self.query_weights)  # (sequence_length, emb_size)
        k = np.matmul(x.T, self.key_weights)    # (sequence_length, emb_size)
        v = np.matmul(x.T, self.value_weights)  # (sequence_length, emb_size)

        # Perform scaled dot-product attention
        output = self.scaled_dot_product_attention(q, k, v)

        return output

def execute(emb_size, num_heads):
    # Load the data
    data = np.random.rand(emb_size, seq_length)
    
    # Initialize the MultiHeadAttention model
    model = Attention(emb_size)
    
    # Perform the forward pass and return the output
    output = model.forward(data, sequence_length=seq_length)
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

