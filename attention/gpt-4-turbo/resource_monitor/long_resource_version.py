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

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

class Attention:
    """
    Implements a simplistic Attention mechanism for educational purposes.
    """
    def __init__(self, emb_size: int):
        """
        Initialize the Attention mechanism.

        Args:
        - emb_size (int): The size of the embeddings (and hence the size of the matrices for Q, K, V).

        """
        self.emb_size = emb_size
        # Initialize weights for Q, K, V
        self.W_q = np.random.rand(emb_size, emb_size)
        self.W_k = np.random.rand(emb_size, emb_size)
        self.W_v = np.random.rand(emb_size, emb_size)
    
    def scaled_dot_product_attention(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray):
        """
        Computes scaled dot-product attention.

        Args:
        - Q (np.ndarray): Queries matrix.
        - K (np.ndarray): Keys matrix.
        - V (np.ndarray): Values matrix.

        Returns:
        - The result of the attention mechanism, and the attention weights.
        """
        d_k = Q.shape[-1]
        
        # Compute the dot product for queries and keys, then scale
        scores = np.dot(Q, K.T) / np.sqrt(d_k)

        # Apply softmax to obtain the weights on the values
        weights = softmax(scores)
        
        # Multiply by V to get the final output
        output = np.dot(weights, V)

        return output, weights

    def forward(self, x: np.ndarray):
        """
        Forward pass to project inputs and compute attention.

        Args:
        - x (np.ndarray): Input tensor of shape (sequence_length, emb_size).

        Returns:
        - The output after applying attention.
        """
        # Check input dimensions
        if x.ndim != 2 or x.shape[1] != self.emb_size:
            raise ValueError(f"Input tensor must be of shape (?, {self.emb_size})")

        # Project the input tensor into Q, K, V
        Q = np.dot(x, self.W_q)
        K = np.dot(x, self.W_k)
        V = np.dot(x, self.W_v)

        # Compute attention
        output, _ = self.scaled_dot_product_attention(Q, K, V)

        return output

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

