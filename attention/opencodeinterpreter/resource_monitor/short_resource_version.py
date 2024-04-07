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

        # Randomly initialize weights for query and value transformations
        self.query = np.random.rand(embedding_size)
        self.value = np.random.rand(embedding_size)

    def forward(self, embeddings):
        '''
        Inputs:
        - embeddings : a sequence of embeddings (sequence_length x embedding_size)

        Returns:
        - outputs : a weighted sum of embeddings (sequence_length x 1)
        '''

        # Dot product between query and each vector in the sequence
        scores = np.matmul(embeddings, self.query)

        # Apply softmax to get weights
        weights = np.exp(scores - np.max(scores)) / np.sum(np.exp(scores - np.max(scores)))

        # Multiply weights with corresponding embeddings and sum them up
        outputs = np.matmul(weights, embeddings)

        return outputs

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

