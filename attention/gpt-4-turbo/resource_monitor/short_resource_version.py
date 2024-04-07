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
        # Initialize weights for query, key, and value matrices
        self.W_query = np.random.rand(embedding_size, embedding_size)
        self.W_key = np.random.rand(embedding_size, embedding_size)
        self.W_value = np.random.rand(embedding_size, embedding_size)
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    
    def forward(self, x):
        """
        x: Input matrix with shape (embedding_size, sequence_length)
        """
        # Compute queries, keys, and values
        queries = np.dot(self.W_query, x)
        keys = np.dot(self.W_key, x)
        values = np.dot(self.W_value, x)

        # Compute attention scores
        # for simplicity, we use dot-product attention here
        attn_scores = np.dot(queries.T, keys)  # Shape: (sequence_length, sequence_length)

        # Apply softmax to get the attention weights
        attn_weights = self.softmax(attn_scores)  # Shape remains the same: (sequence_length, sequence_length)
        
        # Apply the weights to the values
        weighted_values = np.dot(attn_weights, values.T)  # Transpose to get the shape right (sequence_length, embedding_size)
        
        # Here, for simplicity, the output is just the weighted sum of values.
        # In practice, there might be additional operations.
        return weighted_values.T  # Transposing back to maintain consistency (embedding_size, sequence_length)

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

