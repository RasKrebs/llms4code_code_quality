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



def generate_huffman_node_tree(string):
    """Generate a Huffman node tree from a given string.

    Args:
        string (str): Input string for which to create the tree.

    Returns:
        root_node (Node): Root node of the generated Huffman tree.
    """

    # Generate frequency dictionary mapping characters to their frequencies
    freq = {}
    for char in string:
      if char not in freq:
          freq[char] = 1
      else:
          freq[char] += 1

    nodes = sorted(freq.items(), key=lambda x: x[1], reverse=True)

    while len(nodes) > 1:
        left, f1 = nodes[-1]
        right, f2 = nodes[-2]
        nodes = nodes[:-2]

        # Create internal node with character '_' and frequency sum of left and right children
        new_node = Node('_', f1 + f2)
        new_node.left_child = left
        new_node.right_child = right

        if not nodes:
            nodes.append((new_node, f1 + f2))

        # Insert the new node in the sorted list while maintaining sorting order according to frequency
        else:
          for i, (node, freq) in enumerate(nodes):
              if freq <= f1 + f2:
                  nodes.insert(i, (new_node, f1 + f2))
                  break

    root_node = nodes[0][0]
    return root_node

def huffman_code_tree(root_node):
    """Generate a dictionary mapping characters to their Huffman codes.

    Args:
        root_node (Node): Root node of the Huffman tree.

    Returns:
        code_dict (dict): Dictionary mapping characters to their Huffman codes.
    """

    code_dict = {}
    # Traverse the tree recursively and generate huffman code for each character
    def traverse(node, current_code=""):
        if node is None:
            return

        if isinstance(node, Node) and node.character != '_':
                code_dict[node.character] = current_code

        # Recursively traverse the tree and add 0 to huffman codes for left children and 1 for right children
        if isinstance(node, Node):
            traverse(node.left_child, current_code + "0")
            traverse(node.right_child, current_code + "1")

    traverse(root_node)
    return code_dict

class Node:
    """A class representing a node in the Huffman tree."""

    def __init__(self, character, frequency):
        self.character = character  # Character represented by this node (or '_' for internal nodes)
        self.frequency = frequency  # Frequency of the character or subtree represented by this node
        self.left_child = None     # Left child node
        self.right_child = None    # Right child node

# Compression Function:
def compress(input_string):
    """Function to compress a given string using Huffman coding algorithm."""

    # Generate the huffman tree and get the code dictionary
    root_node = generate_huffman_node_tree(input_string)
    code_dict = huffman_code_tree(root_node)

    compressed_output = ""
    
    for char in input_string:
        if char in code_dict:
                compressed_output += code_dict[char]
        else:
                compressed_output += char

    return compressed_output

def execute(text):
    
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

