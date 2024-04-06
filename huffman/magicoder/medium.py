from typing import List, Dict
import heapq

class NodeTree(object):
    def __init__(self, left=None, right=None):
        self.left = left  # type: NodeTree
        self.right = right  # type: NodeTree

class HuffmanCoding():
    @staticmethod
    def calculate_frequency(s: str) -> Dict[str, int]:
        """
        Calculates frequency of each character in the string.

        Parameters:
            s (str): Input string for which frequencies are to be calculated.

        Returns:
            Dictionary with characters as keys and their corresponding frequencies as values.
        """
        freq = {}  # type: Dict[str, int]
        for c in s:
            if c not in freq:
                freq[c] = 0
            freq[c] += 1

        return freq

    @staticmethod
    def generate_heap(freq: Dict[str, int]) -> List[NodeTree]:
        """
        Generates a heap with frequency of each character.

        Parameters:
            freq (Dict): Dictionary containing characters and their frequencies in the string.

        Returns:
            A list representing min-heap that can be used to generate huffman tree.
        """
        heap = [[freq[key], key, NodeTree()] for key in freq]  # type: List[NodeTree]

        heapq.heapify(heap)
        return heap

    @staticmethod
    def build_tree(nodes: List[NodeTree]) -> NodeTree:
        """
        Generates the huffman tree based on the frequencies of each character.

        Parameters:
            nodes (List): A list representing min-heap that can be used to generate huffman tree.

        Returns:
            Root node of the generated Huffman Tree.
        """
        while len(nodes) > 1:
            lo = heapq.heappop(nodes)  # type: List[NodeTree]
            hi = heapq.heappop(nodes)  # type: List[NodeTree]

            left, right = lo[2], hi[2]
            parent_frequency = lo[0] + hi[0]
            parent_node = NodeTree(left=left, right=right)

            heapq.heappush(nodes, [parent_frequency, None, parent_node])

        return nodes[0][2]  # type: NodeTree

    @staticmethod
    def huffman_code_tree(root: NodeTree, s: str = '', mapping: Dict[str, str]=None) -> Dict[str, str]:
        """
        Generates the Huffman Codes for each character.

        Parameters:
            root (NodeTree): Root node of the generated huffman tree.

        Returns:
            Dictionary with characters as keys and their corresponding huffman codes as values.
        """
        if mapping is None:
            mapping = {}  # type: Dict[str, str]

        if root.left:
            HuffmanCoding.huffman_code_tree(root.left, s + '0', mapping)

        if root.right:
            HuffmanCoding.huffman_code_tree(root.right, s + '1', mapping)

        if not root.left and not root.right:
            mapping[root.key] = s  # type: ignore

        return mapping

    @staticmethod
    def compress(s: str) -> str:
        """
        Compresses the input string using huffman encoding.

        Parameters:
            s (str): Input string to be compressed.

        Returns:
            The Huffman encoded string.
        """
        freq = HuffmanCoding.calculate_frequency(s)  # type: Dict[str, int]

        heap = HuffmanCoding.generate_heap(freq)  # type: List[NodeTree]

        root = HuffmanCoding.build_tree(heap)  # type: NodeTree

        codes = HuffmanCoding.huffman_code_tree(root)  # type: Dict[str, str]

        return ''.join([codes[char] for char in s]) + '10'*len(''.join([f"{ord(c):b}" for c in codes]))  # Appended with EOF symbol (10) after all codes to differentiate from the original string

    @staticmethod
    def decompress(s: str) -> str:
        """
        Decompresses the Huffman encoded input string.

        Parameters:
            s (str): Input string in huffman encoding format to be decompressed.

        Returns:
            The original string after being decompressed by the huffman algorithm.
        """
        freq = {}  # type: Dict[str, int]
        while len(s) % 8 != 0:
            s += '1'

        eof_index = -len(''.join([f"{ord(c):b}" for c in codes])) * 2

        codes = HuffmanCoding.huffman_code_tree(HuffmanCoding.build_tree(HuffmanCoding.generate_heap(freq)))  # type: Dict[str, str]

        reverse_mapping = {v: k for k, v in codes.items()}
        return ''.join([reverse_mapping[s[i*8: (i+1)*8]] for i in range(len(s)//8-eof_index//2)])  # type: ignore
