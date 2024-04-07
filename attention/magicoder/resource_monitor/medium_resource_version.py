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
from typing import Tuple

class Attention():
    def __init__(self, emb_size):
        self.emb_size = emb_size  # embedding size

        # generate random weight matrices for query, key, value, and output projections
        self.Wq = np.random.randn(emb_size, emb_size)
        self.Wk = np.random.randn(emb_size, emb_size)
        self.Wv = np.random.randn(emb_size, emb_size)

    def scaled_dot_product_attention(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> Tuple[np.ndarray, float]:
        # Scaled dot product attention
        matmul_qk = np.matmul(Q, K.T)  # shape (seq_len_q, seq_len_v)

        dk = self.emb_size**0.5  # scaling by sqrt(dk)
        scaled_attention_logits = matmul_qk / dk  # softmax is applied internally

        attention_weights = np.exp(scaled_attention_logits)/np.sum(np.exp(scaled_attention_logits), axis=1)

        output = np.matmul(attention_weights, V )  # shape (seq_len_q, emb_size)

        return output, np.maximum(0., scaled_attention_logits).sum()  # for preventing numerical overflow

    def forward(self, input: np.ndarray):

        Q = np.matmul(input, self.Wq)
        K = np.matmul(input, self.Wk)
        V = np.matmul(input, self.Wv)

        output, _ = self.scaled_dot_product_attention(Q, K, V)

        return output  # shape (sequence_length, emb_size)

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

