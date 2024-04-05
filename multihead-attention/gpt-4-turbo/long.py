import numpy as np
import math

class MultiHeadAttention:
    """
    MultiHeadAttention class implements the multi-head attention mechanism
    from the Transformer architecture without using any deep learning frameworks.

    Attributes:
        emb_size (int): The input embedding size.
        num_heads (int): The number of attention heads.
        head_dim (int): Dimension of each attention head.
        weights_query (np.ndarray): Weights for query projection.
        weights_key (np.ndarray): Weights for key projection.
        weights_value (np.ndarray): Weights for value projection.
        weights_out (np.ndarray): Weights for output projection.
    """
    def __init__(self, emb_size: int, num_heads: int):
        """
        Initializes the MultiHeadAttention layer.

        Args:
            emb_size (int): Embedding size of the input.
            num_heads (int): Number of attention heads.
        """
        assert emb_size % num_heads == 0, "emb_size must be divisible by num_heads."
        
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        
        # Initialize weights for query, key, value, and output projection
        self.weights_query = np.random.rand(emb_size, emb_size)
        self.weights_key = np.random.rand(emb_size, emb_size)
        self.weights_value = np.random.rand(emb_size, emb_size)
        self.weights_out = np.random.rand(emb_size, emb_size)

    def scaled_dot_product_attention(self, query: np.ndarray, key: np.ndarray, value: np.ndarray) -> np.ndarray:
        """
        Computes the scaled dot-product attention.

        Args:
            query (np.ndarray): Query tensor.
            key (np.ndarray): Key tensor.
            value (np.ndarray): Value tensor.

        Returns:
            np.ndarray: Result of attention mechanism, scaled dot-product of Q and K, applied softmax and then multiplied by V.
        """
        matmul_qk = np.dot(query, key.transpose(-1, -2))
        
        # Scale the dot-products
        depth = query.shape[-1]
        scaled_attention_logits = matmul_qk / math.sqrt(depth)
        
        # Apply softmax to get probabilities
        attention_weights = self.softmax(scaled_attention_logits)
        
        # Multiply by values
        output = np.dot(attention_weights, value)
        return output

    def softmax(self, x):
        """
        Applies the softmax function.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Softmax-applied output.
        """
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for the MultiHeadAttention layer.

        Args:
            x (np.ndarray): Input tensor of shape (batch_size, seq_length, emb_size).

        Returns:
            np.ndarray: Output tensor after applying multi-head attention.
        """
        batch_size = x.shape[0]
        
        # Project input to query, key, value tensors
        query = np.dot(x, self.weights_query)
        key = np.dot(x, self.weights_key)
        value = np.dot(x, self.weights_value)
        
        # Split tensors into multiple heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        
        # Scaled dot-product attention
        attention = self.scaled_dot_product_attention(query, key, value)
        
        # Concatenate heads
        attention = self.concatenate_heads(attention, batch_size)
        
        # Output projection
        out = np.dot(attention, self.weights_out)
        return out
    
    def split_heads(self, x: np.ndarray, batch_size: int) -> np.ndarray:
        """
        Splits the last dimension of the tensor into (num_heads, head_dim) 
        and transpose the result such that the shape is (batch_size, num_heads, seq_length, head_dim)

        Args:
            x (np.ndarray): Input tensor of shape (batch_size, seq_length, emb_size).
            batch_size (int): Batch size of the input tensor.

        Returns:
            np.ndarray: Tensor with dimensions rearranged to (batch_size, num_heads, seq_length, head_dim).
        """
        x = np.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        return np.transpose(x, (0, 2, 1, 3))

    def concatenate_heads(self, x: np.ndarray, batch_size: int) -> np.ndarray:
        """
        Concatenates the attention heads (inverse of split_heads).

        Args:
            x (np.ndarray): Input tensor of shape (batch_size, num_heads, seq_length, head_dim).
            batch_size (int): Batch size.

        Returns:
            np.ndarray: Concatenated tensor of shape (batch_size, seq_length, emb_size).
        """
        x = np.transpose(x, (0, 2, 1, 3))
        return np.reshape(x, (batch_size, -1, self.emb_size))
