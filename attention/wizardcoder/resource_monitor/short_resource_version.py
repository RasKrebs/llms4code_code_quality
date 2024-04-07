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
    def __init__(self, embedding_size):
        self.embedding_size = embedding_size

        # Initializing weights randomly for simplicity
        self.Wa = np.random.randn(embedding_size, 30) # Weight matrix for attention mechanism
        self.ba = np.random.randn(30, 1)                 # Bias for attention layer

    def forward(self, inputs):
        """
        inputs shape: (embedding_size, sequence_length)
        outputs shape: (sequence_length, sequence_length)
        """
        # Calculate scores via w*x + b
        scores = np.dot(self.Wa.T, inputs) + self.ba

        # Normalize scores with softmax to get attention weights
        alphas = self._softmax(scores)

        return alphas

    def _softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

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

