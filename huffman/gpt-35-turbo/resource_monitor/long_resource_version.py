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



from heapq import heappush, heappop, heapify
from collections import defaultdict

class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_frequency_dict(data: str) -> dict:
    freq_dict = defaultdict(int)
    for char in data:
        freq_dict[char] += 1
    return freq_dict

def build_huffman_tree(freq_dict: dict) -> Node:
    priority_queue = []
    
    for char, freq in freq_dict.items():
        node = Node(char, freq)
        heappush(priority_queue, node)
        
    while len(priority_queue) > 1:
        node1 = heappop(priority_queue)
        node2 = heappop(priority_queue)
        
        merge_node = Node(None, node1.freq + node2.freq)
        merge_node.left = node1
        merge_node.right = node2
        
        heappush(priority_queue, merge_node)
        
    return priority_queue[0]

def build_huffman_codes(node, current_code="", codes={}):
    if node:
        if node.char:
            codes[node.char] = current_code
        build_huffman_codes(node.left, current_code + "0", codes)
        build_huffman_codes(node.right, current_code + "1", codes)
    
    return codes

def huffman_compress(data: str) -> str:
    if not isinstance(data, str):
        raise TypeError("Input data must be a string")
    
    freq_dict = build_frequency_dict(data)
    huffman_tree = build_huffman_tree(freq_dict)
    huffman_codes = build_huffman_codes(huffman_tree)
    
    compressed_data = "".join(huffman_codes[char] for char in data)
    
    return compressed_data

def execute(text):
    root = huffman_compress(text)
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

