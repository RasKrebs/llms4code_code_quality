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



# A Huffman Node
class NodeTree(object):
    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right

    def nodes(self):
        return (self.left, self.right)

# A Huffman Tree
class HuffmanTree(object):
    def __init__(self, root: NodeTree):
        self.root = root

    def traverse_tree(self, node: NodeTree, code=''):
        if type(node) is str:
            return {node: code}
        (l, r) = node.nodes()
        d = dict()
        d.update(self.traverse_tree(l, code + '0'))
        d.update(self.traverse_tree(r, code + '1'))
        return d

def calculate_frequency(text): # 
    """
    Calculate the frequency of each character in the given text.
    """
    frequency = {}
    for character in text:
        if character not in frequency:
            frequency[character] = 1
        else:
            frequency[character] += 1
    return frequency

def build_huffman_tree(frequencies):
    """
    Build the Huffman tree based on the frequencies of characters.
    """
    heap = [[weight, [symbol, ""]] for symbol, weight in frequencies.items()]
    while len(heap) > 1:
        heap.sort()
        left = heap.pop(0)
        right = heap.pop(0)
        for pair in left[1:]:
            pair[1] = '0' + pair[1]
        for pair in right[1:]:
            pair[1] = '1' + pair[1]
        heap.append([left[0] + right[0]] + left[1:] + right[1:])
    return heap[0] if heap else None

def execute(text):
    """
    Execute the Huffman encoding process on the given text.
    """
    # Ensure the input is a string
    if not isinstance(text, str):
        raise ValueError("Input data must be a string.")

    # Calculate frequency of each character
    frequencies = calculate_frequency(text)

    # Build the Huffman tree
    huffman_tree = build_huffman_tree(frequencies)

    # In case huffman_tree is None (empty input text)
    if huffman_tree is None:
        return None

    # Generate Huffman codes
    huffman_codes = {item[0]: item[1] for item in huffman_tree[1:]}

    return huffman_codes



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

