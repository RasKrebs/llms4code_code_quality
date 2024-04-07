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
        Initialize Attention class with input embedding size.

        Args:
        emb_size (int): Size of input embeddings
        """
        self.emb_size = emb_size
        self.Wq = np.random.randn(emb_size, emb_size)
        self.Wk = np.random.randn(emb_size, emb_size)
        self.Wv = np.random.randn(emb_size, emb_size)

    def scaled_dot_product_attention(self, Q: np.array, K: np.array, V: np.array) -> np.array:
        """
        Perform scaled dot-product attention operation.

        Args:
        Q (np.array): Query tensor of shape (emb_size, sequence_length)
        K (np.array): Key tensor of shape (emb_size, sequence_length)
        V (np.array): Value tensor of shape (emb_size, sequence_length)

        Returns:
        np.array: Result of attention operation
        """
        d_k = Q.shape[0]
        att_scores = np.matmul(Q.T, K) / np.sqrt(d_k)
        att_weights = np.exp(att_scores - np.max(att_scores, axis=1, keepdims=True))
        att_weights /= np.sum(att_weights, axis=1, keepdims=True)
        att_output = np.matmul(att_weights, V.T)
        return att_output

    def forward(self, input_tensor: np.array) -> np.array:
        """
        Forward pass of the attention mechanism.

        Args:
        input_tensor (np.array): Input tensor of shape (emb_size, sequence_length)

        Returns:
        np.array: Result of attention operation
        """
        Q = np.matmul(self.Wq, input_tensor)
        K = np.matmul(self.Wk, input_tensor)
        V = np.matmul(self.Wv, input_tensor)
        return self.scaled_dot_product_attention(Q, K, V)

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

