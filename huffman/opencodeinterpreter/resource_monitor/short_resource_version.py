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



class Node:
    def __init__(self, char=None, freq=None, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right

def calculate_frequency(string):
    frequency = {}
    for char in string:
        if char not in frequency:
            frequency[char] = 1
        else:
            frequency[char] += 1
    return frequency

def build_heap(freq):
    heap = []
    for key in freq:
        node = Node(key, freq[key])
        heap.append(node)
    heap.sort(key=lambda x:x.freq)
    return heap

def build_huffman_tree(heap):
    while len(heap) > 1:
        node1 = heap.pop(0)
        node2 = heap.pop(0)
        merged = Node(left=node1, right=node2, freq=node1.freq + node2.freq)
        heap.append(merged)
        heap.sort(key=lambda x:x.freq)
    return heap[0]

def create_huffman_codes(root):
    current_code = ""
    codes = {}

    def traverse(node, code):
        if node is None:
            return
        if node.left is None and node.right is None:
            codes[node.char] = code
        traverse(node.left, code + "0")
        traverse(node.right, code + "1")
    traverse(root, current_code)
    return codes

def huffman_compress(string):
    frequency = calculate_frequency(string)
    heap = build_heap(frequency)
    root = build_huffman_tree(heap)
    codes = create_huffman_codes(root)
    compressed = ""
    for char in string:
        compressed += codes[char]
    return compressed

def execute(text):
    compressed = huffman_compress(text)  # Calculate frequency of each character


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

