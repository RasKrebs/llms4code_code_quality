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



class NodeTree:
    """
    A class to represent a node in a Huffman Code tree.

    Attributes:
        left (NodeTree): The left child of the node.
        right (NodeTree): The right child of the node.

    Methods:
        children(): Get the children of the node.
        __str__(): Get a string representation of the node.
    """

    def __init__(self, left=None, right=None):
        """
        Initialize a NodeTree object.

        Args:
            left (NodeTree): The left child of the node.
            right (NodeTree): The right child of the node.
        """
        self.left = left
        self.right = right

    def children(self):
        """
        Get the children of the node.

        Returns:
            tuple: A tuple containing the left and right children of the node.
        """
        return self.left, self.right


def huffman_code_tree(node, bin_string=''):
    """
    Find the Huffman Code for each character in the tree.

    Args:
        node (NodeTree or str): The current node in the tree.
        bin_string (str): The binary string representation of the code.

    Returns:
        dict: A dictionary containing the Huffman Code for each character.
    """
    # If the node is None, return an empty dictionary
    if node is None:
        return {}

    # If the node is a string, return the character and its code
    if isinstance(node, str):
        return {node: bin_string}

    # Continue traversing the tree
    (left, right) = node.children()

    # Create a dictionary to store the Huffman Code for each character
    dictionary = {}

    # Update the dictionary with the Huffman Code for the left and right children
    dictionary.update(huffman_code_tree(left, bin_string + '0'))
    dictionary.update(huffman_code_tree(right, bin_string + '1'))

    # Return the dictionary
    return dictionary


def make_tree(nodes):
    """
    Make a Huffman Code tree from a list of nodes.

    Args:
        nodes (list): A list of nodes. Each node is a tuple containing a key and a frequency.

    Returns:
        NodeTree: The root of the Huffman Code tree.
    """

    # Continue until there is only one node left in the list
    while len(nodes) > 1:
        # Get the two nodes with the lowest frequencies
        (key1, c_1) = nodes[-1]
        (key2, c_2) = nodes[-2]

        # Remove the two nodes from the list
        nodes = nodes[:-2]

        # Create a new node with the two nodes as children
        node = NodeTree(key1, key2)

        # Calculate the frequency of the new node
        new_frequency = c_1 + c_2

        # Add the new node back to the list
        nodes.append((node, new_frequency))

        # Sort the list in descending order based on frequency
        nodes = sorted(nodes, key=lambda x: x[1], reverse=True)

    # Return the root of the Huffman Code tree
    return nodes[0][0]

def execute(text):
    # Calculate the frequency of each character in the text
    freqs = {}
    for char in text:
        if char in freqs:
            freqs[char] += 1
        else:
            freqs[char] = 1

    # Convert the frequencies dictionary into a list of nodes and create the Huffman tree
    nodes = sorted(freqs.items(), key=lambda x: x[1], reverse=True)
    tree = make_tree(nodes)

    # Generate Huffman codes for each character
    codes = huffman_code_tree(tree)
    return codes



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

