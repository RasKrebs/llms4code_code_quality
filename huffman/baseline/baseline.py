# Description: This file contains the baseline implementation of the Huffman Coding algorithm.
# The code is adapted from the following source: https://favtutor.com/blogs/huffman-coding

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
    # If the node is a string, return the character and its code
    if isinstance(node, str):
        return {node: bin_string}
    # Otherwise, continue traversing the tree
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
