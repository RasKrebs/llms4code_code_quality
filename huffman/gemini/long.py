from collections import Counter
from heapq import heappop, heappush


class Node:
    """
    Node class to represent a node in the Huffman tree.
    """

    def __init__(self, char: str, freq: int, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq


def huffman_code_tree(text: str) -> Node:
    """
    Creates a Huffman tree for the given text.

    Args:
        text: The input text string.

    Returns:
        The root node of the Huffman tree.

    Raises:
        TypeError: If the input is not a string.
    """

    if not isinstance(text, str):
        raise TypeError("Input must be a string")

    # Count character frequencies
    freq_table = Counter(text)

    # Create a priority queue with nodes for each character
    queue = [Node(char, freq) for char, freq in freq_table.items()]
    heapq.heapify(queue)

    # Build Huffman tree
    while len(queue) > 1:
        # Extract two nodes with lowest frequencies
        node1 = heappop(queue)
        node2 = heappop(queue)

        # Create a new internal node with the sum of frequencies
        combined_freq = node1.freq + node2.freq
        parent = Node(None, combined_freq, node1, node2)

        # Push the new node back into the queue
        heappush(queue, parent)

    # Return the root node of the Huffman tree
    return queue[0]


def generate_codes(node: Node, prefix: str, codes: dict) -> None:
    """
    Generates Huffman codes for each character in the tree.

    Args:
        node: The current node in the tree.
        prefix: The current code prefix for the node.
        codes: A dictionary to store character-code mappings.
    """

    if node is None:
        return

    if node.char is not None:
        codes[node.char] = prefix
        return

    generate_codes(node.left, prefix + "0", codes)
    generate_codes(node.right, prefix + "1", codes)


def compress(text: str) -> str:
    """
    Compresses the given text using Huffman coding.

    Args:
        text: The input text string.

    Returns:
        The compressed string representation (encoded codes).

    Raises:
        TypeError: If the input is not a string.
    """

    if not isinstance(text, str):
        raise TypeError("Input must be a string")

    # Build Huffman tree
    tree = huffman_code_tree(text)

    # Generate Huffman codes
    codes = {}
    generate_codes(tree, "", codes)

    # Encode the text using Huffman codes
    compressed = ""
    for char in text:
        compressed += codes[char]

    return compressed


# Example usage
text = "This is an example string to compress."
compressed_text = compress(text)

print("Original text:", text)
print("Compressed text:", compressed_text)
