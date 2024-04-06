import heapq
from collections import defaultdict
from typing import Optional, Union

class Node:
    """Node class for Huffman tree."""

    def __init__(self, char: str, freq: int) -> None:
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    # defining comparators less_than and equals
    def __lt__(self, other):
        return self.freq < other.freq

    def __eq__(self, other):
        return self.freq == other.freq

def huffman_tree(data: str) -> Node:
    """Generate Huffman tree."""
    if not isinstance(data, str):
        raise TypeError("Input should be a string.")

    frequency = defaultdict(int)
    for character in data:
        frequency[character] += 1

    heap = []
    for key in frequency:
        node = Node(key, frequency[key])
        heapq.heappush(heap, node)

    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)

        merged = Node(None, node1.freq + node2.freq)
        merged.left = node1
        merged.right = node2

        heapq.heappush(heap, merged)

    return heap[0]  # root of the Huffman tree

def huffman_code_tree(node: Node, code="", codes={}) -> dict:
    """Generate Huffman codes for each character."""
    if node is None:
        raise ValueError("Empty Huffman tree.")

    # This node is a leaf node then
    if not node.left and not node.right:
        codes[node.char] = code

    if node.left:
        huffman_code_tree(node.left, code + '0', codes)
    if node.right:
        huffman_code_tree(node.right, code + '1', codes)

    return codes

def huffman_encode(data: str) -> Optional[str]:
    """Encode input string using Huffman coding."""
    if not isinstance(data, str):
        raise TypeError("Input should be a string.")

    tree = huffman_tree(data)
    codes = huffman_code_tree(tree)

    encoded_str = ''.join([codes[character] for character in data])

    return encoded_str
