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



import heapq
from collections import defaultdict
from typing import Optional, Union

class Node:
    """Node class for Huffman tree."""

    def __init__(self, char: str, freq: int) -> None:
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    # defining comparators less_than and equals
    def __lt__(self, other):
        return self.freq < other.freq

    def __eq__(self, other):
        return self.freq == other.freq

def huffman_tree(data: str) -> Node:
    """Generate Huffman tree."""
    if not isinstance(data, str):
        raise TypeError("Input should be a string.")

    frequency = defaultdict(int)
    for character in data:
        frequency[character] += 1

    heap = []
    for key in frequency:
        node = Node(key, frequency[key])
        heapq.heappush(heap, node)

    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)

        merged = Node(None, node1.freq + node2.freq)
        merged.left = node1
        merged.right = node2

        heapq.heappush(heap, merged)

    return heap[0]  # root of the Huffman tree

def huffman_code_tree(node: Node, code="", codes={}) -> dict:
    """Generate Huffman codes for each character."""
    if node is None:
        raise ValueError("Empty Huffman tree.")

    # This node is a leaf node then
    if not node.left and not node.right:
        codes[node.char] = code

    if node.left:
        huffman_code_tree(node.left, code + '0', codes)
    if node.right:
        huffman_code_tree(node.right, code + '1', codes)

    return codes

def huffman_encode(data: str) -> Optional[str]:
    """Encode input string using Huffman coding."""
    if not isinstance(data, str):
        raise TypeError("Input should be a string.")

    tree = huffman_tree(data)
    codes = huffman_code_tree(tree)

    encoded_str = ''.join([codes[character] for character in data])

    return encoded_str

def execute(text):
    if not isinstance(text, str):
        raise ValueError("Input data must be a string.")  # Ensure the input is a string
    
    root = huffman_encode(text)  # Generate the Huffman tree from the input text
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

