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



class Attention:
    """Attention layer for transformer models"""

    def __init__(self, embedding_size):
        # Initialize weights with random values
        self.query_weights = np.random.rand(embedding_size, embedding_size)
        self.key_weights = np.random.rand(embedding_size, embedding_size)
        self.value_weights = np.random.rand(embedding_size, embedding_size)
        self.embedding_size = embedding_size

    def scaled_dot_product_attention(self, query: np.ndarray, key: np.ndarray, value: np.ndarray):
        """Calculates the attention weights"""
        # Compute the dot product between query and key matrices
        attn = np.matmul(query, np.transpose(key))

        # Scale by square root of embedding size
        scaled_attn = attn / (self.embedding_size ** 0.5)

        # Apply softmax to normalize the weights
        return Softmax().forward(scaled_attn, axis=-1), value

    def forward(self, x: np.ndarray):
        """Projects input into query, key and values and computes attention"""

        assert len(x.shape) == 2, "Input must be two-dimensional."
        # Project the input tensor using weights
        queries = np.matmul(x.T, self.query_weights)
        keys = np.matmul(x.T, self.key_weights)
        values = np.matmul(x.T, self.value_weights)

        # Calculate the attention using scaled dot product
        attn, values = self.scaled_dot_product_attention(queries, keys, values)

        return attn, values

class Softmax:
    """Softmax activation layer"""

    def __init__(self):
        pass

    def forward(self, x: np.ndarray, axis=0):
        # Subtract the max value for numerical stability
        e_x = np.exp(x - np.expand_dims(np.max(x, axis=axis), axis))

        # Normalize by dividing by the sum along specified axis
        return e_x / np.expand_dims(e_x.sum(axis=axis), axis)

    def backward(self, dvalues: np.ndarray):
        """ Backward pass for softmax function """

        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):

            # Flatten output array
            single_output = single_output.reshape(-1, 1)

            # Calculate Jacobian matrix of the output and
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            # Calculate sample-wise gradient
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

        return self.dinputs

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

