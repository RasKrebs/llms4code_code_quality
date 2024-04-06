from typing import Dict, List, Tuple
import heapq

class Node:
    """
    A class to represent a node in the Huffman tree.

    Attributes:
        char (str): The character represented by the node.
        freq (int): The frequency of the character.
        left (Node): The left child of the node.
        right (Node): The right child of the node.
    """
    def __init__(self, char: str, freq: int, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq

class HuffmanTree:
    """
    A class to represent a Huffman tree.

    Attributes:
        root (Node): The root of the Huffman tree.
    """
    def __init__(self, text: str):
        self.root = self.build_tree(text)

    def build_tree(self, text: str) -> Node:
        """
        Builds the Huffman tree for the given text.

        Args:
            text (str): The text to build the tree for.

        Returns:
            Node: The root of the Huffman tree.
        """
        freq_dict = self.get_frequency(text)
        heap = [Node(char, freq) for char, freq in freq_dict.items()]
        heapq.heapify(heap)

        while len(heap) > 1:
            node1 = heapq.heappop(heap)
            node2 = heapq.heappop(heap)
            merged = Node(None, node1.freq + node2.freq, node1, node2)
            heapq.heappush(heap, merged)

        return heap[0]

    def get_frequency(self, text: str) -> Dict[str, int]:
        """
        Calculates the frequency of each character in the given text.

        Args:
            text (str): The text to calculate the frequency for.

        Returns:
            Dict[str, int]: A dictionary mapping characters to their frequencies.
        """
        freq_dict = {}
        for char in text:
            if char not in freq_dict:
                freq_dict[char] = 0
            freq_dict[char] += 1
        return freq_dict

class HuffmanCoding:
    """
    A class to represent the Huffman coding algorithm.

    Attributes:
        tree (HuffmanTree): The Huffman tree used for encoding and decoding.
    """
    def __init__(self, text: str):
        self.tree = HuffmanTree(text)

    def encode(self, text: str) -> str:
        """
        Encodes the given text using the Huffman tree.

        Args:
            text (str): The text to encode.

        Returns:
            str: The encoded text.
        """
        encoding_dict = self.get_encoding_dict(self.tree.root)
        return ''.join([encoding_dict[char] for char in text])

    def decode(self, encoded_text: str) -> str:
        """
        Decodes the given encoded text using the Huffman tree.

        Args:
            encoded_text (str): The encoded text to decode.

        Returns:
            str: The decoded text.
        """
        decoded_text = []
        node = self.tree.root
        for bit in encoded_text:
            if bit == '0':
                node = node.left
            else:
                node = node.right
            if node.char is not None:
                decoded_text.append(node.char)
                node = self.tree.root
        return ''.join(decoded_text)

    def get_encoding_dict(self, node: Node, path: str = '') -> Dict[str, str]:
        """
        Generates the encoding dictionary for the Huffman tree.

        Args:
            node (Node): The current node in the tree.
            path (str): The path taken to reach the current node.

        Returns:
            Dict[str, str]: A dictionary mapping characters to their Huffman codes.
        """
        if node is None:
            return {}
        if node.char is not None:
            return {node.char: path}
        return {
            **self.get_encoding_dict(node.left, path + '0'),
            **self.get_encoding_dict(node.right, path + '1')
        }
