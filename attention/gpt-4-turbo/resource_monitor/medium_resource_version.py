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
    A simple implementation of the Attention mechanism with Scaled Dot-Product Attention.
    This class initializes random weight matrices for query, key, value, and output projections.
    """

    def __init__(self, emb_size: int):
        """
        Initializes the Attention mechanism.

        Parameters:
        - emb_size (int): The size of the embedding vectors.
        """
        self.emb_size = emb_size
        # Initialize weight matrices for query, key, value, and output projection
        self.W_query = np.random.rand(emb_size, emb_size)
        self.W_key = np.random.rand(emb_size, emb_size)
        self.W_value = np.random.rand(emb_size, emb_size)
        self.W_out = np.random.rand(emb_size, emb_size)

    def scaled_dot_product_attention(self, query: np.ndarray, key: np.ndarray, value: np.ndarray) -> np.ndarray:
        """
        Computes the Scaled Dot-Product Attention.

        Parameters:
        - query, key, value (np.ndarray): Matrices containing the query, key, and value vectors.

        Returns:
        - np.ndarray: The result of applying the attention mechanism.
        """
        d_k = query.shape[1]
        scores = np.matmul(query, key.T) / np.sqrt(d_k)  # Scaled scores
        attn_weights = self.softmax(scores)
        return np.matmul(attn_weights, value)

    def softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the softmax of the input array.

        Parameters:
        - x (np.ndarray): The input array.

        Returns:
        - np.ndarray: Softmax applied to the input array.
        """
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass of the Attention mechanism.

        Parameters:
        - x (np.ndarray): Input matrix of shape (embedding_size, sequence_length).

        Returns:
        - np.ndarray: Output of the Attention mechanism.
        """
        # Project inputs to query/key/value vectors
        queries = np.dot(x, self.W_query)
        keys = np.dot(x, self.W_key)
        values = np.dot(x, self.W_value)

        # Apply Scaled Dot-Product Attention
        attention_out = self.scaled_dot_product_attention(queries, keys, values)

        # Project the output of the attention
        out = np.dot(attention_out, self.W_out)

        return out

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

