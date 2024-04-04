import heapq  # heap data structure for priority queue
from collections import defaultdict   # dictionary subclass that calls a factory function to supply missing values

# A class for nodes in the huffman tree.
class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    # comparators are imported to use heapify function on list of nodes
    def __lt__(self, other):
        return self.freq < other.freq

# Function to calculate frequency for each character and build the huffman tree.
def build_tree(string):

    # Calculate frequency of each character in string.
    freq = defaultdict(int)
    for char in string:
        freq[char] += 1

    # Build heap with all the nodes.
    nodes = [Node(char, freq[char]) for char in freq]
    heapq.heapify(nodes)

    # Loop until only root remains (root of Huffman tree).
    while len(nodes) > 1:
        node1 = heapq.heappop(nodes)
        node2 = heapq.heappop(nodes)

        # Create a new node for the combined weight of two nodes and push it back into list.
        merged_node = Node(None, node1.freq + node2.freq)
        merged_node.left = node1
        merged_node.right = node2

        heapq.heappush(nodes, merged_node)

    return nodes[0] if nodes else None  # Return the root of Huffman tree.

# Function to traverse the huffman tree and store code in a dictionary.
def generate_codes(root, code=""):
    codes = {}
    if root is not None:
        if root.char is not None:  # If leaf node then we found a character.
            codes[root.char] = code

        else:  # Traverse the tree.
            generate_codes(root.left, code + "0")
            generate_codes(root.right, code + "1")

    return codes

# Function to compress string using Huffman coding.
def huffman_compress(string):

    root = build_tree(string)  # Build the Huffman tree.
    codes = generate_codes(root)

    compressed_str = ""
    for char in string:
        compressed_str += codes[char]  # Append respective code to the string.

    return compressed_str, root