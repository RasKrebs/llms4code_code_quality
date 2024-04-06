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

def huffman_compress(string):
    root = build_tree(string)  # Build the Huffman tree.
    codes = generate_codes(root)

    compressed_str = ""
    for char in string:
        if char in codes:
            compressed_str += codes[char]  # Append respective code to the string.
        else:
            continue
            #raise ValueError(f"Character '{char}' is not present in the Huffman codes.")

    return compressed_str, root


def execute(text):
    if not isinstance(text, str):
        raise ValueError("Input data must be a string.")  # Ensure the input is a string
    
    root = huffman_compress(text)  # Generate the Huffman tree from the input text
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

