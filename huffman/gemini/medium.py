from collections import Counter


class Node:
    """
    Node class to represent a node in the Huffman tree.
    """

    def __init__(self, char: str, freq: int, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right

    def __repr__(self):
        return f"Node(char: '{self.char}', freq: {self.freq})"


def huffman_code_tree(text: str) -> Node:
    """
    Builds a Huffman tree from the character frequencies in the text.

    Args:
        text: The input text string.

    Returns:
        The root node of the Huffman tree.
    """

    # Count character frequencies
    char_freq = Counter(text)

    # Create a priority queue of nodes using frequency
    nodes = [Node(char, freq) for char, freq in char_freq.items()]
    from heapq import heappop, heappush  # for min heap operations

    while len(nodes) > 1:
        # Extract two nodes with lowest frequencies
        min1 = heappop(nodes)
        min2 = heappop(nodes)

        # Create a new internal node with the sum of frequencies
        internal_node = Node(None, min1.freq + min2.freq, min1, min2)

        # Push the new node back to the queue
        heappush(nodes, internal_node)

    # The remaining node is the root of the Huffman tree
    return nodes[0]


def generate_codes(node: Node, code: str = "", codes: dict[str, str] = None) -> dict[str, str]:
    """
    Generates Huffman codes for each character by traversing the Huffman tree.

    Args:
        node: The current node in the Huffman tree.
        code: The current code path (0 for left, 1 for right).
        codes: A dictionary to store character-code mappings (optional).

    Returns:
        A dictionary mapping characters to their corresponding Huffman codes.
    """

    if codes is None:
        codes = {}

    if node is None:
        return codes

    # Traverse left subtree with '0' appended to code
    if node.left:
        codes = generate_codes(node.left, code + "0", codes.copy())

    # Traverse right subtree with '1' appended to code
    if node.right:
        codes = generate_codes(node.right, code + "1", codes.copy())

    # If leaf node, add character-code mapping
    if node.char:
        codes[node.char] = code

    return codes


def compress(text: str) -> str:
    """
    Compresses the input text using Huffman coding.

    Args:
        text: The input text string.

    Returns:
        The compressed text string using Huffman codes.
    """

    # Build the Huffman tree
    tree = huffman_code_tree(text)

    # Generate Huffman codes
    codes = generate_codes(tree)

    # Encode text using Huffman codes
    compressed = "".join([codes[char] for char in text])

    return compressed