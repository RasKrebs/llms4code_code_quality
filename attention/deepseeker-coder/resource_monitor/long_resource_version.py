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
    def __init__(self, emb_size: int):
        """
        Initialize the Attention class.

        Parameters:
        emb_size (int): The size of the input embeddings.
        """
        self.emb_size = emb_size
        # Initialize weights with random values
        self.W_q = np.random.randn(emb_size, emb_size)
        self.W_k = np.random.randn(emb_size, emb_size)
        self.W_v = np.random.randn(emb_size, emb_size)

    def scaled_dot_product_attention(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        Perform scaled dot product attention.

        Parameters:
        Q (np.ndarray): Query matrix.
        K (np.ndarray): Key matrix.
        V (np.ndarray): Value matrix.

        Returns:
        np.ndarray: The result of the attention operation.
        """
        d_k = K.shape[1]
        # Compute the dot product of Q and K, and scale by the square root of d_k
        scores = np.matmul(Q, K.T) / np.sqrt(d_k)
        # Apply softmax to get the attention weights
        attn_weights = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
        # Compute the weighted sum of the value vectors
        output = np.matmul(attn_weights, V)
        return output

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Perform the forward pass of the Attention mechanism.

        Parameters:
        x (np.ndarray): Input tensor of shape (sequence_length, embedding_size).

        Returns:
        np.ndarray: The result of the attention operation.
        """
        # Project input into query, key, and value vectors
        Q = np.matmul(x.T, self.W_q)
        K = np.matmul(x.T, self.W_k)
        V = np.matmul(x.T, self.W_v)
        # Perform attention operation
        output = self.scaled_dot_product_attention(Q, K, V)
        return output

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

