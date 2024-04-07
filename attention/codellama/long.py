class Attention:
    """Attention layer for transformer models"""

    def __init__(self, embedding_size):
        # Initialize weights with random values
        self.query_weights = np.random.rand(embedding_size, embedding_size)
        self.key_weights = np.random.rand(embedding_size, embedding_size)
        self.value_weights = np.random.rand(embedding_size, embedding_size)

    def scaled_dot_product_attention(self, query: np.ndarray, key: np.ndarray, value: np.ndarray):
        """Calculates the attention weights"""
        # Compute the dot product between query and key matrices
        attn = np.matmul(query, np.transpose(key))

        # Scale by square root of embedding size
        scaled_attn = attn / (self.embedding_size ** 0.5)

        # Apply softmax to normalize the weights
        return softmax(scaled_attn, axis=-1), value

    def forward(self, x: np.ndarray):
        """Projects input into query, key and values and computes attention"""

        assert len(x.shape) == 2, "Input must be two-dimensional."
        # Project the input tensor using weights
        queries = np.matmul(x, self.query_weights)
        keys = np.matmul(x, self.key_weights)
        values = np.matmul(x, self.value_weights)

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
