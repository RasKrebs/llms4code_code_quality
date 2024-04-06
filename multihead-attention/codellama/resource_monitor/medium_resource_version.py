import psutil
import os
import threading
import multiprocessing

# Define a global variable to store the maximum resources usage
max_resources_usage = {"cpu": 0, "memory": 0}


import numpy as np
import random

np.random.seed(42)
seq_length = 100 
batch_size = 32
emb_size = 512
num_heads = 8 


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



class MultiHeadAttention(object):
    """
    A Multi-head Attention module. Inputs are batched sequences of vectors, which are then
    processed using a scaled dot-product attention mechanism and returned as weighted outputs.
    Parameters:
        emb_size (int): The size of the input embeddings.
        num_heads (int): The number of heads to use for multi-head attention.
       Raises:
           ValueError: If 'emb_size' is not divisible by 'num_heads'.
   """
    def __init__(self, emb_size: int, num_heads: int):
        if emb_size % num_heads != 0:
            raise ValueError("Embedding size (emb_size) should be divisible by the number of heads.")
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        
        # Generate random weight matrices for query, key, value and output projections
        self.Wq = np.random.randn(emb_size, emb_size) / np.sqrt(emb_size)
        self.Wk = np.random.randn(emb_size, emb_size) / np.sqrt(emb_size)
        self.Wv = np.random.randn(emb_size, emb_size) / np.sqrt(emb_size)
        self.Wo = np.random.randn(emb_size, emb_size) / np.sqrt(emb_size)

    def scaled_dot_product_attention(self, query: np.ndarray, key: np.ndarray, value: np.ndarray):
        """
        Calculates the scaled dot-product attention for a batch of input sequences.

           Parameters:
               query (np.ndarray): The query vectors for each sequence element.
               key (np.ndarray): The key vectors for each sequence element.
               value (np.ndarray): The value vectors for each sequence element.

           Returns:
               context_vectors (np.ndarray): The context vectors computed using scaled dot-product attention.
        """

        # Calculate the unscaled dot-product of query and key
        unscaled_dot_product = np.matmul(query, key.transpose())

        # Scale by square root of embedding dimension to prevent numeric overflow
        scaled_dot_product = unscaled_dot_product / np.sqrt(self.emb_size)

        # Apply softmax to ensure attention weights sum to 1
        attention_weights = self.softmax(scaled_dot_product)

        # Calculate the weighted output vectors for each sequence element
        context_vectors = np.matmul(attention_weights, value)

        return context_vectors

    def softmax(self, x: np.ndarray):
        """
        Calculates the softmax of a vector or matrix element-wise.

           Parameters:
               x (np.ndarray): The input vector or matrix.

           Returns:
               softmax_x (np.ndarray): The softmax of the input.
        """

        e_x = np.exp(x)
        sum_e_x = np.sum(e_x, axis=-1).reshape(-1, 1)
        softmax_x = e_x / sum_e_x

        return softmax_x

    def forward(self, x: np.ndarray):
        """
        Calculates the output of the Multi-head Attention module for a batched input sequence.

           Parameters:
               x (np.ndarray): The input sequences as a matrix of shape (batch_size, seq_length, emb_size).

           Returns:
               output (np.ndarray): The weighted output vectors for each sequence element.
        """

        # Split the input into heads along the embedding dimension
        x = x.reshape(x.shape[0], x.shape[1], self.num_heads, -1)

        # Project the queries, keys and values for each head
        query = np.matmul(x, self.Wq)
        key = np.matmul(x, self.Wk)
        value = np.matmul(x, self.Wv)

        # Calculate the attention outputs for each head
        attn_outputs = [self.scaled_dot_product_attention(query[:, :, i], key[:, :, i], value[:, :, i]) for i in range(self.num_heads)]

        # Concatenate the attention outputs of each head into a single vector
        attn_output = np.concatenate(attn_outputs, axis=-1)

        # Project the concatenated output
        output = np.matmul(attn_output, self.Wo)

        return output
def execute(batch_size, emb_size, num_heads, seq_length):
    """
    Executes the MultiHeadAttention model with generated input data.
    
    Args:
        batch_size (int): The number of samples in the batch.
        emb_size (int): The size of each input embedding vector.
        num_heads (int): The number of attention heads.
    
    Returns:
        ndarray: The output from the MultiHeadAttention model.
    """
    # Load the data
    data = np.random.rand(batch_size, seq_length, emb_size)
    
    # Initialize the MultiHeadAttention model
    model = MultiHeadAttention(emb_size=emb_size, num_heads=num_heads)
    
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
    output = execute(batch_size, emb_size, num_heads, seq_length)


    # Stop the monitoring
    monitoring = False
    monitor_thread.join()

    print(max_resources_usage['cpu']), print(max_resources_usage['memory'] / (1024**2))

