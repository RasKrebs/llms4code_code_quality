import heapq
from collections import Counter, deque

# Create a tree node class to store character, frequency, and left and right child nodes
class TreeNode:
    def __init__(self, char=None, freq=0):
        self.char = char
        self.freq = freq
        self.left_child = None
        self.right_child = None

# Function to create Huffman tree based on frequency analysis of characters in the string
def build_huffman_tree(data):
    # Create a frequency table using Counter from collections module
    freq_table = Counter(data)

    # Initialize priority queue with fixed size (equal to length of unique characters in data)
    pq = []
    for char, freq in freq_table.items():
        heapq.heappush(pq, (freq, char))

    # Create a list to store tree nodes ordered by frequency
    node_list = []

    # Merge lowest two frequencies using loop until there is only one node left in the priority queue
    while len(pq) > 1:
        # Pop two lowest frequencies from the priority queue
        freq1, char1 = heapq.heappop(pq)
        freq2, char2 = heapq.heappop(pq)

        # Create a new internal node with these two nodes as children and add it to the priority queue (with updated frequency)
        merged_freq = freq1 + freq2
        merged_node = TreeNode()
        merged_node.left_child = TreeNode(char1, freq1)
        merged_node.right_child = TreeNode(char2, freq2)

        # Push the new internal node into the priority queue and maintain the heap invariant
        heapq.heappush(pq, (merged_freq, merged_node))

    # Return the Huffman tree root node
    return pq[0][1]

# Function to create a dictionary with character as key and its encoded value as value
def create_encoding_map(root):
    if not root:
        return {}  # Return empty dict if root is None (edge case)

    encoding_map = {}
    dq = deque()  # Use a double-ended queue to store nodes and codes

    # Push the root node and an empty code to the front of the queue
    dq.appendleft((root, ""))

    while len(dq) > 0:
        node, code = dq.pop()  # Pop a node and its corresponding code from the front of the queue

        if not node.char:
            # If it's an internal node, push both children to the back of the queue with updated codes (append '0' for left child and '1' for right child)
            dq.append((node.left_child, code + "0"))
            dq.append((node.right_child, code + "1"))
        else:
            # If it's a leaf node, add the character and its code to the encoding map
            encoding_map[node.char] = code

    return encoding_map

# Function to compress input data using Huffman coding (returns compressed string)
def huffman_compress(data):
    # Create Huffman tree based on frequency analysis of characters in the data
    root = build_huffman_tree(data)

    # Create encoding map with characters as keys and their encoded values as values
    encoding_map = create_encoding_map(root)

    compressed = ""
    for char in data:
        compressed += encoding_map[char]  # Add the corresponding Huffman code to the compressed string for each character in the input data

    return compressed
