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



from heapq import heappop, heappush
from collections import defaultdict
import string

class Node(object):
    def __init__(self, left=None, right=None, root=None, val=None):
        self.left = left
        self.right = right
        self.root = root
        self.val = val

def build_tree(node, string, i):
    if node:
        if isinstance(node, str): # If the node value is a character then print it
            string += node
            return string
        else:  # Otherwise, traverse tree
            string = build_tree(node.left, string, '0')
            string = build_tree(node.right, string, '1')
    return string

def huffman_code_tree(text):
    frequency = defaultdict(int) # Dictionary with characters as keys and frequencies as values
    for char in text:
        if isinstance(char, str) and char not in string.punctuation + string.whitespace:
            frequency[char] += 1

    heap = [[weight, [char]] for char, weight in frequency.items()] # Create a min-heap

    while len(heap) > 1:
        lo = heappop(heap)
        hi = heappop(heap)

        for pair in lo[1:] + hi[1:]:
            try:
                pair[1] = '0' + pair[1] if pair[1].startswith('0') else '1' + pair[1]  # Create code
            except:
                continue

        heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    # The root of the Huffman tree is the last item in the heap
    root = heappop(heap)[1][0]

    # Build the Huffman tree and get the codes
    codes = build_tree(root, "", "")

    # Return the codes sorted by length and reversed
    return sorted(codes, key=len)[-1][::-1]

def execute(text):
    if not isinstance(text, str):
        raise ValueError("Input data must be a string.")  # Ensure the input is a string
    
    root = huffman_code_tree(text)  # Generate the Huffman tree from the input text
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

