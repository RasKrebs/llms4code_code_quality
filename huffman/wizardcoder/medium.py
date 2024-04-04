import heapq
from collections import Counter
from typing import Dict, Tuple, Union

class Node:
    """Class representing a node in the Huffman Tree."""
    def __init__(self, char: str, freq: int):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    # defining comparators less_than and equal_to
    def __lt__(self, other):
        return self.freq < other.freq

    def __eq__(self, other):
        return self.freq == other.freq

class HuffmanCoding:
    """Class representing a Huffman Coding."""
    def __init__(self, text: str):
        self.text = text

    def make_frequency_dict(self, text: str) -> Dict[str, int]:
        """Return a dictionary with the frequency of each character in the given text."""
        return dict(Counter(text))

    def build_heap(self, freq_dict: Dict[str, int]) -> Tuple[Node]:
        """Build and return a priority queue based on frequency of characters."""
        heap = []
        for key in freq_dict:
            node = Node(key, freq_dict[key])
            heapq.heappush(heap, node)

        return heap

    def merge_nodes(self, heap: Tuple[Node]) -> Union[Node, None]:
        """Merge nodes and return the root of Huffman Tree."""
        while len(heap) > 1:
            node1 = heapq.heappop(heap)
            node2 = heapq.heappop(heap)

            merged = Node(None, node1.freq + node2.freq)
            merged.left = node1
            merged.right = node2

            heapq.heappush(heap, merged)
        return heap[0] if heap else None

    def make_codes_helper(self, root: Node, current_code: str, codes: Dict[str, str]) -> None:
        """A helper function for building the Huffman code."""
        if root is None:
            return

        if root.char is not None:
            codes[root.char] = current_code

        self.make_codes_helper(root.left, current_code + "0", codes)
        self.make_codes_helper(root.right, current_code + "1", codes)

    def make_codes(self, heap: Tuple[Node]) -> Dict[str, str]:
        """Return the Huffman code for each character."""
        root = self.merge_nodes(heap)
        codes = {}
        current_code = ""
        self.make_codes_helper(root, current_code, codes)

        return codes

    def get_encoded_text(self, text: str, codes: Dict[str, str]) -> str:
        """Return the encoded version of the given text."""
        return ''.join([codes[character] for character in text])

    def huffman_encoding(self) -> Tuple[Dict[str, str], str]:
        freq_dict = self.make_frequency_dict(self.text)
        heap = self.build_heap(freq_dict)
        codes = self.make_codes(heap)
        encoded_text = self.get_encoded_text(self.text, codes)

        return (codes, encoded_text)