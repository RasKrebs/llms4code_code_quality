import math
import random

class MultiHeadAttention(object):
    def __init__(self, embedding_size, num_heads):
        self.embedding_size = embedding_size
        self.num_heads = num_heads

        # Generate random weights for the query, key, and value matrices
        self.Wq = [[random.random() for _ in range(self.embedding_size)] for _ in range(self.embedding_size)]
        self.Wk = [[random.random() for _ in range(self.embedding_size)] for _ in range(self.embedding_size)]
        self.Wv = [[random.random() for _ in range(self.embedding_size)] for _ in range(self.embedding_size)]

    def forward(self, input):
        # Check if the number of heads divides the embedding size evenly
        assert self.num_heads % self.embedding_size == 0, "Number of heads must be divisible by the embedding size"

        # Calculate the depth of each head (embedding_size / num_heads)
        dk = self.embedding_size // self.num_heads

        # Split the input tensor into multiple heads and apply attention
        output = [self.attention(input, input, input) for _ in range(self.num_heads)]

        # Concatenate the outputs of each head
        concat_output = []
        for row in output:
            concat_row = []
            for element in row:
                concat_row += element
            concat_output.append(concat_row)

        return concat_output

    def attention(self, query, key, value):
        # Calculate the query-key product and normalize it with a softmax function
        qk = [[sum([query[i][j] * key[i][j] for j in range(len(key[i]))]) for i in range(len(key))] for i in range(len(value))]
        attn_weights = self.softmax(qk)

        # Apply the attention weights to the value vector and calculate the output
        output = [[sum([attn_weights[i][j] * value[j][k] for j in range(len(value))]) for k in range(len(key))] for i in range(len(query))]

        return output

    def softmax(self, x):
        # Calculate the softmax of a vector
        e_x = [math.exp(i) for i in x]
        sum_e_x = sum(e_x)
        return [i / sum_e_x for i in e_x]