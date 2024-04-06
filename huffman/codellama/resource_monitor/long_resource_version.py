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



import math
from collections import defaultdict
import heapq

# Node class to build the tree
class Node(object):
    def __init__(self, value=None, freq=0.0):
        self.value = value
        self.freq = freq
        self.left_child = None
        self.right_child = None

    # Getter methods for the properties of Node class objects
    def get_value(self):
        return self.value

    def get_freq(self):
        return self.freq

    def get_left_child(self):
        return self.left_child

    def get_right_child(self):
        return self.right_child

    # Setter methods for the properties of Node class objects
    def set_value(self, value):
        self.value = value

    def set_freq(self, freq):
        self.freq = freq

    def set_left_child(self, node):
        self.left_child = node

    def set_right_child(self, node):
        self.right_child = node

    # Comparison methods to support heap operations for Node objects
    def __lt__(self, other):
        return self.freq < other.freq

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.freq == other.freq
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, Node):
            return self.freq <= other.freq
        return NotImplemented

    # Method to check if a node is a leaf (has no children)
    def is_leaf(self):
        return self.left_child == None and self.right_child == None

# Function to generate the huffman_code_tree from input data
def huffman_code_tree(data):
    try:
        # Check if input data is valid (i.e., is a string)
        if not isinstance(data, str):
            raise TypeError("Input data must be a string.")

        # Create a frequency table from the input data
        freq_table = defaultdict(int)
        for char in data:
            freq_table[char] += 1

        # Create a heap to hold the nodes of our Huffman tree
        heap = []

        # Add each character as a leaf node to the heap
        for char, freq in freq_table.items():
            node = Node(value=char, freq=freq)
            heapq.heappush(heap, node)

        # Build the Huffman tree by merging nodes
        while len(heap) > 1:
            # Pop two smallest nodes from the heap
            node1 = heapq.heappop(heap)
            node2 = heapq.heappop(heap)

            # Create a new internal node with these as children
            parent_node = Node()
            parent_node.set_left_child(node1)
            parent_node.set_right_child(node2)
            parent_node.set_freq(node1.get_freq() + node2.get_freq())

            # Add the new internal node to the heap
            heapq.heappush(heap, parent_node)

        # Return the root of the Huffman tree (which is now a single node)
        return heap[0]

    except TypeError as e:
        print(e)  # Handle type error by printing message
        return None

# Function to generate huffman codes from a given Huffman tree
def generate_huffman_codes(node, prefix="", code=None):
    if code is None:  # Initialize empty dictionary for codes
        code = {}

    try:
        # Check if input node is a valid Node object (or has the get_value() method)
        value = node.get_value()

        # If the current node is a leaf, store its code in the dictionary
        if node.is_leaf():
            code[value] = prefix

        # Recursively traverse left and right subtrees to generate codes
        else:
            generate_huffman_codes(node.get_left_child(), prefix + "0", code)  # Left branch is a '0' bit
            generate_huffman_codes(node.get_right_child(), prefix + "1", code)  # Right branch is a '1' bit

        return code

    except AttributeError as e:
        print("Node must be an object of the Node class.")
        return None
    
# Function to encode text using generated Huffman codes
def huffman_encoding(data):
    try:
        # Check if input data is valid (i.e., is a string)
        if not isinstance(data, str):
            raise TypeError("Input data must be a string.")

        # Generate the Huffman tree and codes for the input text
        root = huffman_code_tree(data)
        codes = generate_huffman_codes(root)

        # Encode each character in the input text using its corresponding code
        encoded_text = ""
        for char in data:
            encoded_text += codes[char]

        return encoded_text, root

    except TypeError as e:
        print(e)  # Handle type error by printing message
        return None, None
    
def huffman_decoding(encoded_data, root):
    try:
        # Check if input data is valid (i.e., is a string)
        if not isinstance(encoded_data, str):
            raise TypeError("Input data must be a string.")

        decoded_text = ""  # Initialize an empty string to store the decoded text
        node = root  # Start from the Huffman tree's root node

        for char in encoded_data:
            if char == "0":
                node = node.get_left_child()  # Traverse left subtree

                # Check if we have reached a leaf (decoded character)
                if node.is_leaf():
                    decoded_text += node.get_value()  # Append the decoded character to the text
                    node = root  # Reset current node to the Huffman tree's root
            else:
                node = node.get_right_child()  # Traverse right subtree

                # Check if we have reached a leaf (decoded character)
                if node.is_leaf():
                    decoded_text += node.get_value()  # Append the decoded character to the text
                    node = root  # Reset current node to the Huffman tree's root

        return decoded_text

    except TypeError as e:
        print(e)  # Handle type error by printing message
        return None

def execute(text):
    root = huffman_encoding(text)  # Generate the Huffman tree from the input text
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

