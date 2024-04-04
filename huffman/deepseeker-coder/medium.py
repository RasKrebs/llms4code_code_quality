import heapq
import os
from collections import defaultdict
from typing import Dict, List, Tuple

class NodeTree(object):
    """
    A class to create a node in the Huffman tree.
    """
    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right

    def children(self):
        return (self.left, self.right)

    def nodes(self):
        return (self.left, self.right)

    def __str__(self):
        return '%s_%s' % (self.left, self.right)


def huffman_code_tree(node, binString=''):
    """
    A function to generate Huffman code for each character.
    """
    if type(node) is str:
        return {node: binString}
    (l, r) = node.children()
    d = dict()
    d.update(huffman_code_tree(l, binString + '0'))
    d.update(huffman_code_tree(r, binString + '1'))
    return d


def tree_generator(data: str) -> Tuple[NodeTree, Dict[str, str]]:
    """
    A function to generate the Huffman tree.
    """
    frequency = defaultdict(int)
    for symbol in data:
        frequency[symbol] += 1

    heap = [[weight, [symbol, ""]] for symbol, weight in frequency.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return huffman_code_tree(heap[0][0])


def huffman_encoding(data: str) -> Tuple[str, Dict[str, str]]:
    """
    A function to encode the data using Huffman coding.
    """
    huffman_code = tree_generator(data)
    encoded_output = "".join([huffman_code[char] for char in data])
    return encoded_output, huffman_code


def huffman_decoding(encoded_data: str, huffman_code: Dict[str, str]) -> str:
    """
    A function to decode the data using Huffman coding.
    """
    huffman_code_reverse = {v: k for k, v in huffman_code.items()}
    temp = ""
    decoded_output = ""

    for digit in encoded_data:
        temp += digit
        if temp in huffman_code_reverse:
            character = huffman_code_reverse[temp]
            decoded_output += character
            temp = ""

    return decoded_output