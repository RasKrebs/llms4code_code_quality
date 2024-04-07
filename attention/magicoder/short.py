import numpy as np

class Attention():
    def __init__(self, embedding_size):
        self.W = np.random.rand(embedding_size, embedding_size)  # Weight matrix for computing attention weights

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def forward(self, h):
        u = np.matmul(h, self.W)  # Compute unnormalized attention scores
        a = self.softmax(u)  # Compute the softmax of the normalized attention scores

        return a
