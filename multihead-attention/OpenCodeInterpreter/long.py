import math
from typing import Tuple

class MultiHeadAttention:
    def __init__(self, emb_size: int, num_heads: int):
        assert emb_size % num_heads == 0, "emb_size should be divisible by num_heads"

        self.emb_size = emb_size
        self.num_heads = num_heads
        self.depth = emb_size // num_heads

        # Initialize weights for query, key, value, and output projections
        self.query = [[math.random()] * self.emb_size for _ in range(self.num_heads)]
        self.key = [[math.random()] * self.emb_size for _ in range(self.num_heads)]
        self.value = [[math.random()] * self.emb_size for _ in range(self.num_heads)]
        self.output = [[math.random()] * emb_size for _ in range(self.num_heads)]

    def scaled_dot_product_attention(self, query: Tuple[int], key: Tuple[int], value: Tuple[int]) -> Tuple[float]:
        dot_product = sum(q * k for q, k in zip(query, key)) / math.sqrt(len(query))
        softmax = lambda x : math.exp(x) / sum(math.exp(i) for i in x)

        attention_weights = [softmax(dot_product)] * len(value)

        weighted_sum = [w * v for w, v in zip(attention_weights, value)]

        return tuple(weighted_sum)

    def forward(self, input: Tuple[Tuple[int]]) -> Tuple[float]:
        # Project input into query, key, and value tensors using the initialized weights
        queries = [tuple(sum(q * i for q, i in zip(query_weight, input_)) for input_ in input) for query_weight in self.query]
        keys = [tuple(sum(k * i for k, i in zip(key_weight, input_)) for input_ in input) for key_weight in self.key]
        values = [tuple(sum(v * i for v, i in zip(value_weight, input_)) for input_ in input) for value_weight in self.value]

        # Apply attention operation to each head
        attention_outputs = []
        for query_, key_, value_ in zip(queries, keys, values):
            attention_result = self.scaled_dot_product_attention(query_, key_, value_)
            attention_outputs.append(attention_result)

        # Concatenate the results from each head and apply an output projection
        concat_attention = [sum(i[j] for i in attention_outputs) for j in range(len(input[0]))]
        output = tuple(sum(o * i for o, i in zip(output_weight, concat_attention)) for output_weight in self.output)

        return output
