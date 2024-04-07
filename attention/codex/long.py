import math
import random

class Attention:
    """
    Attention class for transformer models.
    """
    def __init__(self, emb_size: int):
        """
        Initialize with emb_size (input embedding size).
        Initialize weights with random values.
        """
        self.emb_size = emb_size
        self.weights = [random.random() for _ in range(emb_size)]

    def scaled_dot_product_attention(self, query, key, value):
        """
        Perform the attention operation, performing scaled_dot_product,
        which also applied softmax for stability.
        """
        matmul_qk = sum([q * k for q, k in zip(query, key)])

        # scale matmul_qk
        dk = math.sqrt(len(query))
        scaled_attention_logits = matmul_qk / dk

        # softmax is applied for stability
        attention_weights = math.exp(scaled_attention_logits) / sum(math.exp(x) for x in [query, key, value])

        output = sum([v * aw for v, aw in zip(value, attention_weights)])

        return output, attention_weights

    def forward(self, input_tensor):
        """
        The forward method projects input tensors into query, key, and value tensors
        using the initialized weights and performs attention.
        """
        query = [i * w for i, w in zip(input_tensor, self.weights)]
        key = [i * w for i, w in zip(input_tensor, self.weights)]
        value = [i * w for i, w in zip(input_tensor, self.weights)]

        return self.scaled_dot_product_attention(query, key, value)
