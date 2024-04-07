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
    Attention mechanism implementation from scratch in Python.
    """
    def __init__(self, emb_size: int, seq_len: int):
        """
        Initialize the Attention mechanism with the given embedding size and sequence length.
        Random weight matrices for query, key, value, and output projections are generated.
        """
        self.emb_size = emb_size
        self.seq_len = seq_len
        self.query_weights = np.random.rand(emb_size, emb_size)
        self.key_weights = np.random.rand(emb_size, emb_size)
        self.value_weights = np.random.rand(emb_size, emb_size)
        self.output_weights = np.random.rand(emb_size, emb_size)

    def scaled_dot_product_attention(self, query: np.ndarray, key: np.ndarray, value: np.ndarray):
        """
        Scaled dot-product attention mechanism.
        It can handle numeric overflow.
        """
        matmul_qk = np.matmul(query, np.transpose(key))

        # scale matmul_qk
        depth = key.shape[-1]
        logits = matmul_qk / np.sqrt(depth)

        # softmax is applied to axis -1 so that the scores
        # add up to 1.
        attention_weights = self.softmax(logits, axis=-1)

        output = np.matmul(attention_weights, value)

        return output, attention_weights

    def softmax(self, x: np.ndarray, axis=None):
        """
        Compute softmax values for each sets of scores in x.
        """
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / e_x.sum(axis=axis, keepdims=True)

    def forward(self, x: np.ndarray):
        """
        Forward method performs the computation and returns the output.
        """
        query = np.matmul(x.T, self.query_weights)
        key = np.matmul(x.T, self.key_weights)
        value = np.matmul(x.T, self.value_weights)

        out, _ = self.scaled_dot_product_attention(query, key, value)
        out = np.matmul(out, self.output_weights)

        return out

def execute(emb_size, num_heads):
    # Load the data
    data = np.random.rand(emb_size, seq_length)
    
    # Initialize the MultiHeadAttention model
    model = Attention(emb_size, seq_len=seq_length)
    
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

