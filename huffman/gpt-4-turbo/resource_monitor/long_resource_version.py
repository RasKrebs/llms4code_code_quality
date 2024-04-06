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



from typing import Dict, Tuple
from collections import Counter, deque
import heapq

class Node:
    def __init__(self, char: str, freq: int):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None
    
    # For priority queue, defines comparison operators based on frequency
    def __lt__(self, other):
        return self.freq < other.freq

    def __eq__(self, other):
        if other is None:
            return False
        if not isinstance(other, Node):
            raise TypeError(f"'==' not supported between instances of 'Node' and '{type(other)}'")
        return self.freq == other.freq

def build_tree(text: str) -> Node:
    """
    Build Huffman Tree from text.
    
    Args:
    text (str): The input text of which the Huffman Tree is to be constructed.
    
    Returns:
    Node: The root node of the constructed Huffman Tree.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")
    
    # Count frequency of appearance for each character
    freq = Counter(text)
    if not freq:
        raise ValueError("Input string cannot be empty.")
    
    # Priority queue for the frequency of appearance
    priority_queue = [Node(char, freq) for char, freq in freq.items()]
    heapq.heapify(priority_queue)
    
    # Construct the Huffman Tree
    while len(priority_queue) > 1:
        # Pop two nodes with the smallest frequencies
        left = heapq.heappop(priority_queue)
        right = heapq.heappop(priority_queue)
        
        # Merge nodes
        merged = Node(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        
        # Push the merged node back into the queue
        heapq.heappush(priority_queue, merged)
    
    # The root node
    return priority_queue[0]

def make_codes(root: Node, current_code: str = "", codes: Dict[str, str] = {}) -> Dict[str, str]:
    """
    Generate Huffman codes for characters based on the constructed Huffman Tree.
    
    Args:
    root (Node): The root node of the Huffman Tree.
    current_code (str): The current path code.
    codes (Dict[str, str]): Dictionary holding the codes for each character.
    
    Returns:
    Dict[str, str]: A dictionary with characters as keys and their corresponding codes as values.
    """
    if root is None:
        return {}

    # If it's a leaf node, add the character and its code to the dictionary
    if root.char is not None:
        codes[root.char] = current_code

    make_codes(root.left, current_code + "0", codes)
    make_codes(root.right, current_code + "1", codes)
    
    return codes

def huffman_encoding(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Encodes a text using Huffman coding.
    
    Args:
    text (str): The text to be encoded.
    
    Returns:
    Tuple[str, Dict[str, str]]: A tuple containing the encoded text and a dictionary of Huffman codes.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")
    
    root = build_tree(text)
    codes = make_codes(root)
    encoded_text = ''.join(codes[char] for char in text)
    
    return encoded_text, codes

def huffman_decoding(encoded_text: str, codes: Dict[str, str]) -> str:
    """
    Decodes an encoded text using Huffman codes.
    
    Args:
    encoded_text (str): The encoded text.
    codes (Dict[str, str]): The dictionary of Huffman codes.
    
    Returns:
    str: The decoded original text.
    """
    # Reverse the code dictionary for decoding: from code to character
    reverse_codes = {v: k for k, v in codes.items()}
    decoded_text = ""
    current_code = ""
    for bit in encoded_text:
        current_code += bit
        if current_code in reverse_codes:
            decoded_text += reverse_codes[current_code]
            current_code = ""  # reset for next character
    return decoded_text

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

