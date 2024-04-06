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

class Node(object):
    def __init__(self, character=None, frequency=0, left=None, right=None):
        self.character = character
        self.frequency = frequency
        self.left = left
        self.right = right

    def is_leaf(self):
        return not (self.right or self.left)

    # Define __lt__ for heapq comparison
    def __lt__(self, other):
        return self.frequency < other.frequency

def huffman_coding(string):
    frequency = defaultdict(int)
    for char in string:
        frequency[char] += 1

    heap = [Node(character=char, frequency=freq) for char, freq in frequency.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)

        merged_node = Node(frequency=node1.frequency+node2.frequency, left=node1, right=node2)

        heapq.heappush(heap, merged_node)

    root = heap[0]

    # Create Huffman Code
    huffman_code = {}
    def create_huffman_code(root, current_bits):
        if root is None:
            return

        if root.is_leaf():
            huffman_code[root.character] = ''.join(current_bits)

        current_bits.append('0')
        create_huffman_code(root.left, current_bits)
        current_bits.pop()

        current_bits.append('1')
        create_huffman_code(root.right, current_bits)
        current_bits.pop()

    create_huffman_code(root, [])

    # Encode string
    encoded_string = ''.join([huffman_code[char] for char in string])

    return huffman_code, encoded_string

def execute(text):
    if not isinstance(text, str):
        raise ValueError("Input data must be a string.")  # Ensure the input is a string
    
    root = huffman_coding(text)  # Generate the Huffman tree from the input text
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

