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



import random
import numpy as np

class Attention(object):
    def __init__(self, embedding_size):
        """Initialization function for the attention layer.

        Args:
            embedding_size (int): Embedding size for the input sequence.
        """
        self.embedding_size = embedding_size

        # Initialize random weights
        self.weights = np.random.randn(self.embedding_size) / np.sqrt(self.embedding_size)

    def forward(self, x):
        """Forward pass for the attention layer.

        Args:
            x (numpy array): Input sequence of shape (sequence_length, embedding_size).

        Returns:
            context_vector (numpy array): Context vector after applying attention.
        """
        # Calculate scores and apply softmax
        scores = np.dot(self.weights, x.T)   # shape: (1, sequence_length)
        alpha = self._softmax(scores)        # shape: (1, sequence_length)

        # Compute context vector as weighted average of input vectors
        context_vector = np.sum(alpha * x.T, axis=0)  # shape: (embedding_size,)

        return context_vector

    def _softmax(self, x):
        """Softmax function for applying attention weights."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

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

