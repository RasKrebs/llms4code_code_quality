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
