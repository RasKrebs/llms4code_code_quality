from typing import Dict, List, Tuple
from queue import PriorityQueue

class NodeTree(object):
    """
    A node tree which will be used while creating the Huffman tree.
    """
    def __init__(self, char: str, freq: int):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

class HuffmanCodeTree(object):
    """
    The Huffman Code Tree.
    """
    def __init__(self, frequency: Dict[str, int]):
        self.frequency = frequency
        self.codes = {}

    def build(self):
        """
        Build the Huffman tree and generate the Huffman codes.
        """
        pqueue = PriorityQueue()

        for char, freq in self.frequency.items():
            pqueue.put((freq, NodeTree(char, freq)))

        while pqueue.qsize() != 1:
            node1 = pqueue.get()[1]
            node2 = pqueue.get()[1]

            merged = NodeTree(None, node1.freq + node2.freq)
            merged.left = node1
            merged.right = node2

            pqueue.put((merged.freq, merged))

        root = pqueue.get()[1]
        self._generate_codes(root, "")

    def _generate_codes(self, node: NodeTree, current_code: str):
        """
        Generate the Huffman codes.
        """
        if node is None:
            return

        if node.char is not None:
            self.codes[node.char] = current_code

        self._generate_codes(node.left, current_code + "0")
        self._generate_codes(node.right, current_code + "1")

    def get_codes(self) -> Dict[str, str]:
        """
        Get the generated Huffman codes.
        """
        return self.codes

def compress(data: str) -> Tuple[str, Dict[str, str]]:
    """
    Compress the input data using Huffman coding.
    """
    if not isinstance(data, str):
        raise ValueError("Input data must be a string")

    frequency = {}
    for char in data:
        if char not in frequency:
            frequency[char] = 0
        frequency[char] += 1

    huffman_tree = HuffmanCodeTree(frequency)
    huffman_tree.build()

    codes = huffman_tree.get_codes()

    compressed_data = ""
    for char in data:
        compressed_data += codes[char]

    return compressed_data, codes