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
        # Initializing weight matrices for query, key, value and output projections
        self.wq = np.random.randn(embedding_size, embedding_size)
        self.wk = np.random.randn(embedding_size, embedding_size)
        self.wv = np.random.randn(embedding_size, embedding_size)
        self.wo = np.random.randn(embedding_size, embedding_size)

    def scaled_dot_product_attention(self, query, keys):
        # Calculating the scaled dot-product attention between query and key vectors
        scores = np.matmul(query.T, keys) / np.sqrt(keys.shape[1])

        # Applying softmax function to calculate weights for each value vector
        weights = self._softmax(scores)

        # Calculating the context vector by taking weighted sum of value vectors
        context_vector = np.matmul(weights, keys.T)

        return context_vector, weights

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))  # For numeric stability
        return e_x / e_x.sum()

    def forward(self, sequence):
        if len(sequence) == 0:
            raise ValueError("Sequence cannot be empty.")

        # Projecting the input sequence using query, key and value weight matrices
        q = np.matmul(sequence.T, self.wq)  # Transpose self.wq
        k = np.matmul(sequence.T, self.wk)  # Transpose self.wk
        v = np.matmul(sequence.T, self.wv)  # Transpose self.wv

        # Calculating the scaled dot-product attention and passing it through output projection layer
        context_vector, attention_weights = self.scaled_dot_product_attention(q, k)
        output = np.matmul(context_vector.T, self.wo)

        return output, attention_weights

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

