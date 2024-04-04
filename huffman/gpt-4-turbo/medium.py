import heapq
from collections import defaultdict, Counter
from typing import Dict, Tuple

class NodeTree:
    """
    A class representing a node in the Huffman tree.

    Attributes:
        char (str): The character this node represents.
        freq (int): The frequency of occurrence of the character.
        left (NodeTree): The left child node.
        right (NodeTree): The right child node.
    """

    def __init__(self, char: str, freq: int):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    # This makes the node class compatible with heapq by comparing frequencies.
    def __lt__(self, other):
        return self.freq < other.freq

def generate_huffman_tree(frequencies: Dict[str, int]) -> NodeTree:
    """
    Generates a Huffman tree based on character frequencies.

    Args:
        frequencies (Dict[str, int]): A dictionary where keys are characters
                                      and values are the frequencies of the characters.

    Returns:
        NodeTree: The root node of the Huffman tree.
    """
    priority_queue = [NodeTree(char, freq) for char, freq in frequencies.items()]
    heapq.heapify(priority_queue)
    
    while len(priority_queue) > 1:
        left = heapq.heappop(priority_queue)
        right = heapq.heappop(priority_queue)
        merged = NodeTree(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(priority_queue, merged)
    
    return priority_queue[0]

def huffman_code_tree(node: NodeTree, prefix: str="", code: Dict[str, str]={}) -> Dict[str, str]:
    """
    Generates Huffman codes for each character by traversing the Huffman tree.

    Args:
        node (NodeTree): The current node in the tree.
        prefix (str): The current prefix code.
        code (Dict[str, str]): The dictionary to store the character codes.
    
    Returns:
        Dict[str, str]: A dictionary of Huffman codes for each character.
    """
    if node is not None:
        if node.char is not None:
            code[node.char] = prefix
        huffman_code_tree(node.left, prefix+"0", code)
        huffman_code_tree(node.right, prefix+"1", code)
    return code

def compress_string(input_string: str) -> str:
    """
    Compresses a string using Huffman coding.

    Args:
        input_string (str): The string to compress.

    Returns:
        str: A compressed string represented as a sequence of Huffman codes.
    """
    # Calculate character frequency and generate Huffman tree.
    frequencies = dict(Counter(input_string))
    root = generate_huffman_tree(frequencies)
    
    # Generate Huffman codes for characters
    huffman_codes = huffman_code_tree(root)

    # Encode the input string
    encoded_string = ''.join(huffman_codes[char] for char in input_string)
    
    return encoded_string
