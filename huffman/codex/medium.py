from typing import Dict, List, Tuple
from queue import PriorityQueue

class NodeTree(object):
    """
    A NodeTree object has a character, frequency, left child, and right child.
    """
    def __init__(self, char: str, freq: int):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        """
        Nodes are compared based on their frequency for priority queue.
        """
        return self.freq < other.freq

class HuffmanCodeTree(object):
    """
    A HuffmanCodeTree object has methods to build a Huffman tree and generate codes.
    """
    def __init__(self, char_freq: Dict[str, int]):
        self.char_freq = char_freq
        self.codes = {}

    def build_tree(self):
        """
        Builds a Huffman tree and returns the root.
        """
        pqueue = PriorityQueue()

        for char, freq in self.char_freq.items():
            node = NodeTree(char, freq)
            pqueue.put(node)

        while pqueue.qsize() > 1:
            left = pqueue.get()
            right = pqueue.get()

            merged = NodeTree(None, left.freq + right.freq)
            merged.left = left
            merged.right = right

            pqueue.put(merged)

        return pqueue.get()

    def generate_codes_helper(self, node: NodeTree, current_code: str):
        """
        Helper function to generate Huffman codes.
        """
        if node is None:
            return

        if node.char is not None:
            self.codes[node.char] = current_code

        self.generate_codes_helper(node.left, current_code + "0")
        self.generate_codes_helper(node.right, current_code + "1")

    def generate_codes(self, node: NodeTree):
        """
        Generates Huffman codes given the root of the Huffman tree.
        """
        current_code = ""
        self.generate_codes_helper(node, current_code)

def compress(data: str) -> Tuple[Dict[str, str], str]:
    """
    Compresses the string and returns a tuple of the Huffman codes and the compressed data.
    """
    # Calculate frequency of each character
    char_freq = {}
    for char in data:
        if char not in char_freq:
            char_freq[char] = 0
        char_freq[char] += 1

    # Build Huffman tree and generate codes
    huffman_tree = HuffmanCodeTree(char_freq)
    root = huffman_tree.build_tree()
    huffman_tree.generate_codes(root)

    # Compress the data
    compressed_data = ""
    for char in data:
        compressed_data += huffman_tree.codes[char]

    return huffman_tree.codes, compressed_data