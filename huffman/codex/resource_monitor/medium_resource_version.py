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



from typing import Dict, List, Tuple
from queue import PriorityQueue

class NodeTree(object):
    """
    A NodeTree object has a character, frequency, left child, and right child.
    """
    def __init__(self, char: str, freq: int):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        """
        Nodes are compared based on their frequency for priority queue.
        """
        return self.freq < other.freq

class HuffmanCodeTree(object):
    """
    A HuffmanCodeTree object has methods to build a Huffman tree and generate codes.
    """
    def __init__(self, char_freq: Dict[str, int]):
        self.char_freq = char_freq
        self.codes = {}

    def build_tree(self):
        """
        Builds a Huffman tree and returns the root.
        """
        pqueue = PriorityQueue()

        for char, freq in self.char_freq.items():
            node = NodeTree(char, freq)
            pqueue.put(node)

        while pqueue.qsize() > 1:
            left = pqueue.get()
            right = pqueue.get()

            merged = NodeTree(None, left.freq + right.freq)
            merged.left = left
            merged.right = right

            pqueue.put(merged)

        return pqueue.get()

    def generate_codes_helper(self, node: NodeTree, current_code: str):
        """
        Helper function to generate Huffman codes.
        """
        if node is None:
            return

        if node.char is not None:
            self.codes[node.char] = current_code

        self.generate_codes_helper(node.left, current_code + "0")
        self.generate_codes_helper(node.right, current_code + "1")

    def generate_codes(self, node: NodeTree):
        """
        Generates Huffman codes given the root of the Huffman tree.
        """
        current_code = ""
        self.generate_codes_helper(node, current_code)

def compress(data: str) -> Tuple[Dict[str, str], str]:
    """
    Compresses the string and returns a tuple of the Huffman codes and the compressed data.
    """
    # Calculate frequency of each character
    char_freq = {}
    for char in data:
        if char not in char_freq:
            char_freq[char] = 0
        char_freq[char] += 1

    # Build Huffman tree and generate codes
    huffman_tree = HuffmanCodeTree(char_freq)
    root = huffman_tree.build_tree()
    huffman_tree.generate_codes(root)

    # Compress the data
    compressed_data = ""
    for char in data:
        compressed_data += huffman_tree.codes[char]

    return huffman_tree.codes, compressed_data

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

