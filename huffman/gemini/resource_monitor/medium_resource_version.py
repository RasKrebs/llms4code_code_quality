import psutil
import os
import threading
import multiprocessing

# Define a global variable to store the maximum resources usage
max_resources_usage = {"cpu": 0, "memory": 0}


import requests

book_url = "https://www.gutenberg.org/cache/epub/1513/pg1513-images.html"
response = requests.get(book_url)
text = response.text


# Implement the resource monitor
def resource_monitor():
    """
    Monitors the CPU and memory usage of the current process, updating global max usage.
    """
    global max_resources_usage
    process = psutil.Process(os.getpid())
    
    while monitoring:
        cpu_usage = process.cpu_percent(interval=1) / multiprocessing.cpu_count()
        memory_usage = process.memory_info().rss
        max_resources_usage['cpu'] = max(max_resources_usage['cpu'], cpu_usage)
        max_resources_usage['memory'] = max(max_resources_usage['memory'], memory_usage)



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

    def __lt__(self, other):
        return self.freq < other.freq


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

def execute(text):
    if not isinstance(text, str):
        raise ValueError("Input data must be a string.")  # Ensure the input is a string
    
    root = compress(text)  # Generate the Huffman tree from the input text
    return root



if __name__ == "__main__":
    # Start the resource monitoring in a separate thread
    global monitoring
    monitoring = True
    monitor_thread = threading.Thread(target=resource_monitor)
    monitor_thread.start()

    # Execute the Huffman coding process
    huffman_codes = execute(text)


    # Stop the monitoring
    monitoring = False
    monitor_thread.join()

    print(max_resources_usage['cpu']), print(max_resources_usage['memory'] / (1024**2))

