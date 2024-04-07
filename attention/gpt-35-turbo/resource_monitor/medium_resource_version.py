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
        Initialize the Attention module with random weight matrices for query, key, value, and output projections.
        
        Args:
        emb_size: int - The size of the input embeddings
        
        """
        self.emb_size = emb_size
        
        # Random weight matrices for query, key, value, and output projections
        self.W_q = np.random.randn(emb_size, emb_size)
        self.W_k = np.random.randn(emb_size, emb_size)
        self.W_v = np.random.randn(emb_size, emb_size)
        self.W_o = np.random.randn(emb_size, emb_size)
        
    def softmax(self, x):
        """
        Perform softmax operation on the input array.
        
        Args:
        x: np.array - Input array
        
        Returns:
        np.array - Softmax output
        """
        exp_x = np.exp(x - np.max(x))  # Avoid numerical instability
        return exp_x / exp_x.sum(axis=0)
    
    def scaled_dot_product_attention(self, Q, K, V):
        """
        Perform scaled dot product attention on query, key, and value matrices.
        
        Args:
        Q: np.array - Query matrix of shape (emb_size, sequence_length)
        K: np.array - Key matrix of shape (emb_size, sequence_length)
        V: np.array - Value matrix of shape (emb_size, sequence_length)
        
        Returns:
        np.array - Output matrix after attention computation
        """
        dk = self.emb_size
        scores = np.matmul(Q.T, K) / np.sqrt(dk)
        attention_weights = self.softmax(scores)
        output = np.matmul(V, attention_weights.T)
        
        return output
    
    def forward(self, input_embeddings):
        """
        Perform the attention computation and return the output.
        
        Args:
        input_embeddings: np.array - Input embeddings of shape (embedding_size, sequence_length)
        
        Returns:
        np.array - Output matrix after attention computation
        """
        Q = np.matmul(self.W_q, input_embeddings)
        K = np.matmul(self.W_k, input_embeddings)
        V = np.matmul(self.W_v, input_embeddings)
        
        output = self.scaled_dot_product_attention(Q, K, V)
        output = np.matmul(self.W_o, output)
        
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

